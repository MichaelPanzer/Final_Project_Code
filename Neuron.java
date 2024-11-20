public class Neuron {
    private float weight, deltaWeight;
    private Node parent, target;

    public Neuron (float weight, Node parent, Node target){
        this.weight = weight;
        this.parent = parent;
        this.target = target;
        deltaWeight = 0;

        parent.addOuptutNeuron(this);
        target.addInputNeuron(this);
    }

    public void removeConnections(){
        parent = null;
        target = null;
        weight = 0;
    }

    public void changeWeight(float difference){
        weight += difference;
    }

    public float getWeight(){
        return weight;
    
    }
    public Node getParent(){
        return parent;
    }
    public Node getTarget(){
        return target;
    }

    public void adjustWeight(){
        weight += deltaWeight;
        deltaWeight = 0;
    }
    public void adjDeltaWeight(float changeInDeltaWeight){
        this.deltaWeight += changeInDeltaWeight;
    }
}
