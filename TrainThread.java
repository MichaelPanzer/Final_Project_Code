public class TrainThread extends Thread{
    private Network net;
    private float[] data;
    private byte label;

    
    public TrainThread(Network net, float[] data, byte label){
        this.net = net;
        this.data = data;
        this.label = label;
    }


    @Override
    public void run(){
        net.run(data);

        synchronized(net){
            net.backPropagate(makeOutputArr(label, 10));
        }
    }

    private float[] makeOutputArr(byte ans, int length){
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



}