public class SigmoidNode extends Node{
    public SigmoidNode(){
        super();
    }

    public SigmoidNode(float bias){
        super(bias);
    }


    @Override public void setActivation(float value){
        this.dActivation = sigmoidPrime(value);
        this.activation = sigmoid(value);
    }


    private float sigmoid(float input){
        float inverseSigmoid = (float)(1 + Math.pow(Math.E, -1*input));
        return 1/inverseSigmoid;
    }
    private float sigmoidPrime(float input){
        float inverseDir = (float)( Math.pow(Math.E, input) * (1 + Math.pow(Math.E, -1*input))*(1 + Math.pow(Math.E, -1*input)) );
        return 1/inverseDir;
    }
    
}
