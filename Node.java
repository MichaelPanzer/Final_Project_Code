import java.util.LinkedList;

public class Node {
    private float bias, deltaBias;
    protected float dActivation, activation;
    private LinkedList<Neuron> ouptutNeurons;
    private LinkedList<Neuron> inputNeurons;

    public Node(){
        dActivation = 0.0f;
        activation = 0.0f;
        bias = 0.0f;

        ouptutNeurons = new LinkedList<Neuron>();
        inputNeurons = new LinkedList<Neuron>();
    }

    public Node(float bias){
        dActivation = 0.0f;
        activation = 0.0f;
        this.bias = bias;

        ouptutNeurons = new LinkedList<Neuron>();
        inputNeurons = new LinkedList<Neuron>();
    }

    public void addOuptutNeuron(Neuron ouptutNeuron){
        ouptutNeurons.add(ouptutNeuron);
    }

    public void addInputNeuron(Neuron inputNeuron){
        inputNeurons.add(inputNeuron);
    }

    public void drop(){
        activation = 0;
        dActivation = 0;
    }

    


    
    
    
    public void setActivation(float value){
        this.dActivation = leakyReLuPrime(value);
        this.activation = leakyReLu(value);
    }
    
    private float leakyReLu(float input){
        if(input < 0){
            return /*0.001f*input*/ 0.0f;
        } else{
            return input;
        }
    }
    private float leakyReLuPrime(float input){
        if(input < 0){
            return 0.0f;
        } else{
            return 1.0f;
        }
    }





    public float getActivation(){//value after being pushed through the sigmoid function
        return activation;
    }
    public float getDActivation(){
        return dActivation;
    }

    public float getBias(){
        return bias;
    }

    public LinkedList<Neuron> getOutNeurons(){
        return ouptutNeurons;
    }
    public LinkedList<Neuron> getInNeurons(){
        return inputNeurons;
    }

    public void adjustBias(){
        bias += deltaBias;
        deltaBias = 0;
    }
    public void adjDeltaBias(float changeInDeltaBias){
        this.deltaBias += changeInDeltaBias;
    }
}
