import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

//This file is for running a previously created network
//File loaded is test_network.idx

public class MnistTest {
    
    public static final int maxRegions = 5;

    public static void main(String[] args){
        
        Network net = null;

        //THE FILENAME IS THE SAVED NETWORK THAT THE PROGRAM RUNS
        try {
            net = new Network(new File("test_network.idx"));
        } catch (IOException e1) {
            System.err.println(e1.getMessage());
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
         


        //Run network on test set

        int corAns = 0;

        for(int i=0;i<testImages.length; i++){
            if (checkAns(net.run(testImages[i]), testLabels[i])){
                corAns ++;
            }
        }

        System.out.println("\n\n" + corAns + "/" + testImages.length);
        
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
            //Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (black), 2maxRegionsmaxRegions means foreground (white).
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

            /* 
            //adding the number of enclosed regions to the end of the array
            int encRegions = getEnclosedRegions(curImage, 28, 28, -127f);

            

            for(int k=0; j<output[i].length; j++,k++){
                if(k == encRegions){
                    output[i][j] = 128f;
                } else{
                    output[i][j] = -127f;
                }
            }

            */

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