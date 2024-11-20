import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

//This file is for training a network on the minst data set
//file saved as network.idx

public class Mnist {

    private static final float initalDropoutRate = 0.3f;

    private static final float initalLearningSpeed = 0.02f;
    private static final float finalLearningSpeed = 0.000000001f;
    
    public static void main(String[] args){
        
        Network net = new Network(new int[]{784, 36, 30, 24, 10}, initalLearningSpeed, 0.0f);
        
        //get training images and labels
        float[][] trainImages = new float[0][0];
        byte[] trainLabels = new byte[0];
        try{
            trainImages = getImages("mnist/train-images.idx3-ubyte");
            trainLabels = getLabels("mnist/train-labels.idx1-ubyte");
        } catch(IOException ex){
            System.err.println(ex.getMessage());
        }

        //Get test images and labels
        float[][] testImages = new float[0][0];
        byte[] testLabels = new byte[0];
        try{
            testImages = getImages("mnist/t10k-images.idx3-ubyte");
            testLabels = getLabels("mnist/t10k-labels.idx1-ubyte");
        } catch(IOException ex){
            System.err.println(ex.getMessage());
        }




        long initTime = System.currentTimeMillis();

        int corAns = 0;

        //Train network
        int epochs = 2;
        int trainingSetLen = trainLabels.length*epochs;
        float invTrainingSetLen = 1/trainingSetLen;
        for(int i=0;i<trainingSetLen; i++){
            int trainExampleIndex = i%trainLabels.length;

            net.run(trainImages[trainExampleIndex], initalDropoutRate - (1 - trainExampleIndex*invTrainingSetLen));//run the actual network

            
            //Bckpropagate to find the gradient of the cost function
            net.backPropagate(makeOutputArr(trainLabels[trainExampleIndex], 11));
            
            net.setLearningSpeed(initalLearningSpeed - (initalLearningSpeed - finalLearningSpeed)*trainExampleIndex*invTrainingSetLen);

            //this whole thing is kinda jank but it works
            if(i%5 == 0){
                net.applyChanges();
            }

            //test the network every 1000 triaining images
            if(i%5000 == 0){
                corAns = testNetwork(testImages, testLabels, net);

                double trainTime = (System.currentTimeMillis()-initTime)/1000.0;

                System.out.println("\n\n" + corAns + "/" + testImages.length + " at " + i + "   Time = " + trainTime + " seconds");

            }
            
        }

        //save network
        try {
            net.writeToFile(new File("network.idx"));
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        
        double trainTime = (System.currentTimeMillis()-initTime)/1000.0;
        System.out.println(trainTime); 

        
         


        //Run network on test set
        corAns = testNetwork(testImages, testLabels, net);

        System.out.println("\n\n" + corAns + "/" + testImages.length + " at " + 60000 + "   Time = " + trainTime + " seconds");
        
    }

    private static int testNetwork(float[][] testImages, byte[] testLabels, Network network){
        int corAns = 0;
        for(int i=0;i<testImages.length; i++){
            if (checkAns(network.run(testImages[i]), testLabels[i])){
                corAns ++;
            }
        }

        return corAns;
        

    }

    private static float[][] getImages(String filePath) throws IOException{

        Path tempPath = Paths.get(filePath);
        byte[] test = Files.readAllBytes(tempPath);
        ByteBuffer buffer1 = ByteBuffer.wrap(test);

        Integer.toString(buffer1.getInt());

        int numImages = buffer1.getInt();

        int numRows = buffer1.getInt();
        int numCols = buffer1.getInt();

        int imageSize = numCols*numRows;

        float[][] output = new float[numImages][imageSize];

        //System.out.println("first 50 regions:");
        
        for(int i=0; i<numImages; i++){

            float[] curImage = new float[imageSize];//cur image is just used to find the number of enclosed regions

            int j=0;
            for(j = 0; j<curImage.length; j++){

                byte b = buffer1.get();

                //fix overflow error and make pixel value from -127,128
                if(b < 0){
                    curImage[j] = -1f*b;
                    output[i][j] = -1f*b;
                } else{
                    curImage[j] = b-127f;
                    output[i][j] = b-127f;
                }



            }


        }


        return output;
    } 

    private static byte[] getLabels(String filePath) throws IOException{


        Path tempPath = Paths.get(filePath);
        byte[] test = Files.readAllBytes(tempPath);
        ByteBuffer buffer1 = ByteBuffer.wrap(test);

        Integer.toString(buffer1.getInt());

        int numLabels = buffer1.getInt();

        byte[] output = new byte[numLabels];


        for(int i=0; i<numLabels; i++){
            output[i] = buffer1.get();

        }

        return output;
    }

    private static float[] makeOutputArr(byte ans, int length){
        float[] output = new float[length];

        for(int i=0; i<output.length; i++){
            if(i == ans){
                output[i] = 1;
            } else{
                output[i] = 0;
            }
        }

        return output;
    }

    private static boolean checkAns(float[] output, int ans){
        float highestValue = output[0];
        int highestIndex = 0;

        for(int i=1; i<output.length; i++){
            if(output[i] > highestValue){
                highestIndex = i;
                highestValue = output[i];
            }
        }

        if(highestIndex == ans){
            return true;
        } else{
            return false;
        }
    }


}
