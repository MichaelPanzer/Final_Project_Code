import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;


public class Network implements Serializable{
    private Node[][] nodeLayers;


    private Neuron[][][] neuronLayers;

    private float learningSpeed;
    private int paramCount;

    public Network(int[] format, float learningSpeed, float dropThreshold){
        this.learningSpeed = learningSpeed;

        paramCount = format[0];
        for(int i=1; i<format.length; i++){
            paramCount += format[i-1] * format[i];//neurons in a specific layer
            paramCount += format[i];//nodes in a specific layer
        }

        //NODE LAYERS ARE FORMATTED[LAYER][NODE]
        nodeLayers = new Node[format.length][];

        //Set input layer to sigmoid activation
        nodeLayers[0] = new Node[format[0]];
        for(int j=0; j<nodeLayers[0].length; j++){
            nodeLayers[0][j] = new SigmoidNode();
        }

        for(int i=1; i<format.length-1; i++){
            nodeLayers[i] = new Node[format[i]];
            for(int j=0; j<nodeLayers[i].length; j++){
                nodeLayers[i][j] = new Node();
            }
        }

        //Set output layer to sigmoid activation
        int finalIndex = format.length-1;
        nodeLayers[finalIndex] = new Node[format[finalIndex]];
        for(int j=0; j<nodeLayers[finalIndex].length; j++){
            nodeLayers[finalIndex][j] = new SigmoidNode();
        }


        //NEURON ARRYAS ARE FORMATED[LAYER][PARENT][TARGET]
        neuronLayers = new Neuron[format.length-1][][];

        //Create Neuron Layers
        for(int i=0; i<format.length-1; i++){//loop through all layers(i is the current parent layer) 
            neuronLayers[i] = new Neuron[format[i]][format[i+1]];
            for(int j=0; j<format[i]; j++){//loop through parent layer(j is the parent layer index)
                for(int k=0; k<format[i+1]; k++){//loop through target layer(k is the target layer index)

                    neuronLayers[i][j][k] = new Neuron((float)(Math.random()-0.5f), nodeLayers[i][j], nodeLayers[i+1][k]);

                }
            }
        }
    }


    public Network(File f) throws IOException{
        learningSpeed = 0.0f;

        Path tempPath = f.toPath();
        byte[] test = Files.readAllBytes(tempPath);
        ByteBuffer buffer = ByteBuffer.wrap(test);


        int nodeLayerCount =  buffer.get();
        //int[] format = new int[nodeLayerCount];

        nodeLayers = new Node[nodeLayerCount][];
        for(int i=0; i<nodeLayerCount; i++){
            nodeLayers[i] = new Node[buffer.getInt()];
        }

        //Make node array
        //NODE LAYERS ARE FORMATTED[LAYER][NODE]

        //Set input layer to sigmoid activation
        for(int j=0; j<nodeLayers[0].length; j++){
            nodeLayers[0][j] = new SigmoidNode(buffer.getFloat());
        }

        //Set hidden layers to ReLu
        for(int i=1; i<nodeLayers.length-1; i++){
            for(int j=0; j<nodeLayers[i].length; j++){
                nodeLayers[i][j] = new Node(buffer.getFloat());
            }
        }

        //Set output layer to sigmoid activation
        int finalIndex = nodeLayers.length-1;
        for(int j=0; j<nodeLayers[finalIndex].length; j++){
            nodeLayers[finalIndex][j] = new SigmoidNode(buffer.getFloat());
        }


        


        //Make neuron array
        //NEURON ARRYAS ARE FORMATED[LAYER][PARENT][TARGET]
        neuronLayers = new Neuron[nodeLayers.length-1][][];

        //Create Neuron Layers
        for(int i=0; i<nodeLayers.length-1; i++){//loop through all layers(i is the current parent layer) 
            neuronLayers[i] = new Neuron[nodeLayers[i].length][nodeLayers[i+1].length];

            for(int j=0; j<neuronLayers[i].length; j++){//loop through parent layer(j is the parent layer index)
                for(int k=0; k<neuronLayers[i][j].length; k++){//loop through target layer(k is the target layer index)
                    neuronLayers[i][j][k] = new Neuron(buffer.getFloat(), nodeLayers[i][j], nodeLayers[i+1][k]);

                }
            }
        }
        
    }


    public void setLearningSpeed(float learningSpeed){
        this.learningSpeed = learningSpeed;
    }

    public float getLearningSpeed(){
        return learningSpeed;
    }

    public float[] run(float[] input){
        //set the values in the first node layer
        for(int i=0; i<input.length; i++){
            this.nodeLayers[0][i].setActivation(input[i]);
        }

        for(int i=0; i<neuronLayers.length; i++){//loops through the neuron layers
            runLayer(neuronLayers[i]);
        }

        //convert the ouptut array of nodes to an array of floats
        float[] output = new float[nodeLayers[nodeLayers.length-1].length];
        for(int i=0; i<output.length; i++){
            output[i] = nodeLayers[nodeLayers.length-1][i].getActivation();
        }

        return output;//the output is the final nodeLayer
    }


    private void runLayer(Neuron[][] neurons){
        for(int i=0; i<neurons[0].length; i++){//i is the index of the target node 

            //dot product of the parent values and neuron weights corresponding with one node in the output array
            float sum = 0.0f;

            for(int j=0; j<neurons.length; j++){//j is the index of the parent node
                sum += neurons[j][i].getWeight() * neurons[j][i].getParent().getActivation();
            }

            sum += neurons[0][i].getTarget().getBias();//adjusts the sum with the bias of the target node


            neurons[0][i].getTarget().setActivation(sum);
            
        }

    }

    public float[] run(float[] input, float dropoutRate){
        //set the values in the first node layer
        for(int i=0; i<input.length; i++){
            this.nodeLayers[0][i].setActivation(input[i]);
        }

        for(int i=0; i<neuronLayers.length-1; i++){//loops through the neuron layers
            float invOneMinusDropout = 1/(1.0f - dropoutRate);//used to scale activation for dropout
            runLayer(neuronLayers[i], dropoutRate, invOneMinusDropout);
        }

        runLayer(neuronLayers[neuronLayers.length-1]);//run the final layer without dropout because we dont want to drop output nodes

        //convert the ouptut array of nodes to an array of floats
        float[] output = new float[nodeLayers[nodeLayers.length-1].length];
        for(int i=0; i<output.length; i++){
            output[i] = nodeLayers[nodeLayers.length-1][i].getActivation();
        }

        return output;//the output is the final nodeLayer
    }


    private void runLayer(Neuron[][] neurons, float dropoutRate, float invOneMinusDropout){
        for(int i=0; i<neurons[0].length; i++){//i is the index of the target node 

            //dot product of the parent values with the neuron weights corresponding with one node in the output array
            float sum = 0.0f;

            for(int j=0; j<neurons.length; j++){//j is the index of the parent node
                sum += neurons[j][i].getWeight()*neurons[j][i].getParent().getActivation();
            }

            sum += neurons[0][i].getTarget().getBias();//adjusts the sum with the bias of the target node

            //droput a random number of nodes based on the dropout rate
            if(Math.random() < dropoutRate){
                neurons[0][i].getTarget().drop();;
            } else{
                neurons[0][i].getTarget().setActivation(sum * invOneMinusDropout);;
            }
            
        }

    }


    public float backPropagate(float[] desiredOutput){
        float cost = 0;
        for(int i=0; i<nodeLayers[nodeLayers.length-1].length; i++){//output nodes
            Node curNode = nodeLayers[nodeLayers.length-1][i];
            cost += (curNode.getActivation() - desiredOutput[i])*(curNode.getActivation() - desiredOutput[i]);

            float dCostDAct = -2 * (curNode.getActivation()-desiredOutput[i]);

            //calculate and sum the delta biases for all nodes (activation and weight product starts at one and is updated recursively)
            calcDeltaBiases(curNode, dCostDAct, 1, 1);

        }

        return cost;
    }
    
    public void applyChanges(){
        //apply biases
        for(int i=0; i<nodeLayers.length; i++){
            for(int j=0; j<nodeLayers[i].length; j++){
                nodeLayers[i][j].adjustBias();
            }
        }

        //apply weights
        for(int i=0; i<neuronLayers.length; i++){//loop through all layers(i is the current parent layer) 
            for(int j=0; j<neuronLayers[i].length; j++){//loop through parent layer(j is the parent layer index)
                for(int k=0; k<neuronLayers[i][j].length; k++){//loop through target layer(k is the target layer index)
                    //stops null pointer exception
                    if(neuronLayers[i][j][k] != null){
                        neuronLayers[i][j][k].adjustWeight();
                    }
                }
            }
        }
    }

    private void calcDeltaBiases(Node node, float dCostdDctivation, float weightProduct, float derActivaitonProduct){
        
        derActivaitonProduct *= node.getDActivation();

        float biasDerivative = dCostdDctivation * derActivaitonProduct * weightProduct;

        node.adjDeltaBias(biasDerivative * learningSpeed);//Adjusts Biases

        //if there is a layer above the current layer then call calcDeltaBiases for the higher layer
        if(node.getInNeurons().size() > 0){
            for(Neuron curNeuron : node.getInNeurons()){//loop throgh all the input neurons of the node


                float weightDerivative = dCostdDctivation * derActivaitonProduct * weightProduct * curNeuron.getParent().getActivation();

                curNeuron.adjDeltaWeight(weightDerivative * learningSpeed);//Adjusts Weights

                //recursivley call calcDeltaBiases for all parents of the input neurons in order to sum the change from every path
                if(Math.abs(curNeuron.getParent().getActivation()) > 1E-4){
                    calcDeltaBiases(curNeuron.getParent(), dCostdDctivation, weightProduct*curNeuron.getWeight(), derActivaitonProduct);
                }
            }
        }
    }

    public void dropout(float dropoutRate){
        for(int i=0; i<nodeLayers.length; i++){
            for(int j=0; j>nodeLayers[i].length; j++){
                if(Math.random() < dropoutRate){
                    nodeLayers[i][j].drop();
                }
            }   
        }
    }


    @Override
    public String toString() {

        String output = "";
        /*
        for(int i=0; i<nodeLayers.length; i++){
            output += nodeLayers[i].length+",";
        }
        */
        output += nodeLayers.length + "\n";

        output += "\nNode Layers:\n";
        for(int i=0; i<nodeLayers.length; i++){;
            output += i + ":";
            for(int j=0; j<nodeLayers[i].length; j++){
                output += (int)(100*nodeLayers[i][j].getBias()) + ",   ";
                output += nodeLayers[i][j].getBias() + ",";

            }
            output += "\n";
        }

        

        output += "NeuronLayers:\n";

        for(int i=0; i<neuronLayers.length; i++){//loop through all layers(i is the current parent layer) 
            output += "NeuronLayer" + i + ":\n";
            for(int j=0; j<neuronLayers[i].length; j++){//loop through parent layer(j is the target layer index)
                output += j + ":";

                for(int k=0; k<neuronLayers[i][j].length; k++){//loop through target layer(k is the parent layer index)
                    output += neuronLayers[i][j][k].getWeight()+ ",";

                }
                output += "\n";
            }
        }
        

        return output;
    }


    public void writeToFile(File file) throws IOException{
        //create a byte buffer that can hold all the paramater values as well as the format array data
        ByteBuffer buffer = ByteBuffer.allocate(paramCount*4 + nodeLayers.length*4 + 1);

        //length of the format array
        buffer.put((byte)nodeLayers.length);
        //format array 
        for(int i=0; i<nodeLayers.length; i++){
            buffer = buffer.putInt(nodeLayers[i].length);
        }

        //Node bias
        for(int i=0; i<nodeLayers.length; i++){;
            //output += i + ":";
            for(int j=0; j<nodeLayers[i].length; j++){
                //output += (int)(100*nodeLayers[i][j].getBias()) + ",   ";
                buffer = buffer.putFloat(nodeLayers[i][j].getBias());

            }
        }

        //Neuron weights
        for(int i=0; i<neuronLayers.length; i++){//loop through all layers(i is the current parent layer) 
            //output += "NeuronLayer" + i + ":\n";
            for(int j=0; j<neuronLayers[i].length; j++){//loop through parent layer(j is the target layer index)
                //output += j + ":";

                for(int k=0; k<neuronLayers[i][j].length; k++){//loop through target layer(k is the parent layer index)
                    buffer = buffer.putFloat(neuronLayers[i][j][k].getWeight());
                    //System.out.println(k);

                }
            }
        }


        Files.write(file.toPath(), buffer.array());
    }



}
