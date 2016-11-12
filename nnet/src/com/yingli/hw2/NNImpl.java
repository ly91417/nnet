package com.yingli.hw2;
/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */
import weka.core.*;
import java.util.*;
public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the input layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public Node outputNodes=null;// list of the output layer nodes
	
	public Instances trainingSet=null;//the training set
	
	double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	public NNImpl(Instances trainingSet, int hiddenNodeCount, double learningRate, int maxEpoch, 
			double [][]hiddenWeights, double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount = trainingSet.get(0).numAttributes()-1;
		int outputNodeCount = 1;//for this project the output node only have one node
		for(int i = 0; i < inputNodeCount; i++)
		{
			Node node = new Node(0);//type 0 stand for input
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);//bias to hidden
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i = 0; i < hiddenNodeCount; i++)// i stand for index of hidden node
		{
			Node node=new Node(2);//hidden node
			//Connecting hidden layer nodes with input layer nodes
			for(int j = 0; j < inputNodes.size(); j++)// j stand for index of input
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);//bias from hidden to output
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)// i stand for index of output node
		{
			Node node=new Node(4);//output
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)// j stand for index of hidden node
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		// TODO: add code here
		//set up all the input values
		ArrayList<Double> input = inst.attributes;
		for (int i =0; i< input.size(); i++) {
			inputNodes.get(i).setInput(input.get(i));
		}
		for (Node hiddenNode : hiddenNodes) {
			hiddenNode.calculateOutput();
		}
		for (Node outputNode : outputNodes) {
			outputNode.calculateOutput();
		}
		double [] output = new double[outputNodes.size()];
		for (int i =0; i < outputNodes.size(); i++) {
			output[i] = outputNodes.get(i).getOutput();
		}
		double max =-1;
		int maxIndex =-1;
		for (int i =0; i < output.length; i++) {
			if (output[i] >= max) {
				max = inst.attributes.get(i);
				maxIndex = i;
			}
		} 
		assert (maxIndex !=-1);
		return maxIndex;
	}

	public void backPropLearning (ArrayList<Instance> trainingSet, NNImpl network) {
		int loop =this.maxEpoch;
		while (loop>=0) {
			loop--;
			for (Instance inst : trainingSet) {
				ArrayList<Double> input = inst.attributes;
				for (int i =0; i< input.size(); i++) {
					inputNodes.get(i).setInput(input.get(i));
				}
				for (Node hiddenNode : hiddenNodes) {
					hiddenNode.calculateOutput();
				}
				for (Node outputNode : outputNodes) {
					outputNode.calculateOutput();
				}
				double [] output = new double[outputNodes.size()];
				for (int i =0; i < outputNodes.size(); i++) {
					output[i] = outputNodes.get(i).getOutput();
				}
				double [] deltaOutput = new double[outputNodes.size()];
				for (int i =0; i< outputNodes.size(); i++) {
					double x = outputNodes.get(i).getSum();
					deltaOutput[i] = (x <= 0? 0:1)*inst.classValues.get(i)-output[i];
				}

				double [] deltaHidden = new double[hiddenNodes.size()];
				for (int i=0 ; i < hiddenNodes.size(); i++) {
					Node hiddenNode = hiddenNodes.get(i);
					double wmd=0;
					for (int j=0; j<outputNodes.size(); j++) {
						Node outputNode = outputNodes.get(j);
						List<NodeWeightPair> pairs = outputNode.parents;
						for (NodeWeightPair np : pairs) {
							if (np.node.equals(hiddenNode)) {
								double temp = np.weight;
								wmd +=temp*deltaOutput[j];
							}
						}
					}
					double ini = hiddenNode.getSum();
					deltaHidden [i] = (ini <=0 ? 0:1) *wmd;
				}
				int j=0;
				for (Node outputNode : outputNodes) {
					List<NodeWeightPair> pairs =  outputNode.parents;
					for (NodeWeightPair pair : pairs) {
						pair.weight += learningRate * outputNode.getSum() * deltaOutput[j];//2. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!getSum or getOutput ?????
					}
					j++;
				}
				int i=0;
				for (Node hiddenNode : hiddenNodes) {
					List<NodeWeightPair> pairs =  hiddenNode.parents;
					if (pairs!=null) { 
						for (NodeWeightPair pair : pairs) {
							pair.weight += learningRate * hiddenNode.getSum() * deltaHidden[i];//3. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!getSum or getOutput ?????\
						}
						
					}
					i++;
				}

			}
		}	
	}
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		// TODO: add code here
		backPropLearning(this.trainingSet,this);
	}
}
