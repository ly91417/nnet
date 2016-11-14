package com.yingli.hw2;
/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */
import weka.core.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
public class NNImpl{
	public boolean withHiddenNode = true;
	public ArrayList<Node> mInputNodeList = new ArrayList<>();
	//list of the input layer nodes
	public ArrayList<ArrayList<Node>> inputNodes = new ArrayList<>();
	//list of the hidden layer nodes
	public ArrayList<Node> hiddenNodes = new ArrayList<>();
	//list of the output layer nodes
	public ArrayList<Node> outputNodes = null;
	//the training set
	public Instances trainingSet = null;
	//variable to store the learning rate
	double learningRate=1.0; 
	//variable to store the maximum number of epochs
	int maxEpoch=1; 
	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */
	public NNImpl(Instances trainingSet, int hiddenNodeCount, double learningRate, int maxEpoch) {
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		if(hiddenNodeCount == 0 ) {
			withHiddenNode = false;
		}
		//how many lists used for input nodes
		int inputNodeCount = trainingSet.get(0).numAttributes() - 1;
		//input layer nodes initially set with inputNodeCount number of Nodes with the biased node
		inputNodes = new ArrayList<ArrayList<Node>>(inputNodeCount + 1);
		for(int i = 0; i < inputNodeCount; i++) {
			ArrayList<Node> inputNodeList = new ArrayList<Node>();
			if(trainingSet.attribute(i).isNominal()) {
				for(int j = 0 ; j < trainingSet.attribute(i).numValues(); j++) {
					Node node = new Node(0);
					mInputNodeList.add(node);
					inputNodeList.add(node);
				}
			}else if(trainingSet.attribute(i).isNumeric()) {
				Node node = new Node(0);
				mInputNodeList.add(node);
				inputNodeList.add(node);
			}
			inputNodes.add(inputNodeList);
		}
		//bias to hidden
		Node biasToHidden = new Node(1);
		ArrayList<Node> biasToHiddenList = new ArrayList<Node>();
		biasToHiddenList.add(biasToHidden);
		mInputNodeList.add(biasToHidden);
		inputNodes.add(biasToHiddenList);
		if(hiddenNodeCount != 0) {
			//hidden layer nodes
			hiddenNodes = new ArrayList<Node> (hiddenNodeCount + 1);
			// i stand for index of hidden node
			for(int i = 0; i < hiddenNodeCount; i++)
			{
				//hidden node
				Node node = new Node(2);
				Random generator = new Random(hiddenNodeCount);
				//for each attribute
				for(int j = 0; j < inputNodes.size(); j++)
				{	
					int number = generator.nextInt(201)-100;
					double initWt = number / 100.0;
					//for each value of the attribute
					for(Node n : inputNodes.get(i)) {
						NodeWeightPair nwp = 
								new NodeWeightPair(n,initWt);
						node.parents.add(nwp);
					}
				}
				hiddenNodes.add(node);
			}
			//bias from hidden to output
			Node biasToOutput=new Node(3);
			hiddenNodes.add(biasToOutput);
		}
		//Output node layer
		Node node=new Node(4);//output
		Random generator = new Random(hiddenNodeCount);
		if(hiddenNodeCount != 0) {
			//Connecting output layer nodes with hidden layer nodes
			for(int j = 0;j < hiddenNodes.size(); j++)// j stand for index of hidden node
			{
				int number = generator.nextInt(201)-100;
				double initWt = number / 100.0;
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), initWt);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}else {
			for(Node input :  this.mInputNodeList) {
				int number = generator.nextInt(201)-100;
				double initWt = number / 100.0;
				NodeWeightPair nwp = new NodeWeightPair(input, initWt);
				node.parents.add(nwp);
			}
		}
	}
	/**
	 * getSigmoidalOutput
	 */
	private static double getSigmoidalOutput(double value) {
		return (1.0 / (1.0 + Math.pow(Math.E, -value)));
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
		//set up all the input values
		for(int i =0 ; i < trainingSet.numAttributes()-1; i++ ) {	
			//if the attribute is nominal set up the null values 0 and the value in position 1 using 1 of k encoding
			if(inst.attribute(i).isNominal()) {
				double in = inst.value(i);
				int num = inst.numValues();
				double [] inputDouble = new double[num];
				inputDouble[(int)in] = 1;
				List<Node> list = inputNodes.get(i);
				for(int j = 0 ; j < num; j++ ){
					list.get(j).setInput(inputDouble[j]);
				}
			}
			//if it is numerical standardize the input double and then setInput
			else if(inst.attribute(i).isNumeric()) {
				double inputDouble = standardizeInput(inst, i);
				inputNodes.get(i).get(0).setInput(inputDouble);
			}
		}
		if(withHiddenNode) {
			for(Node hiddenNodeH : hiddenNodes) {
				hiddenNodeH.calculateOutput();
			}
			outputNodes.get(0).calculateOutput();
		}
		else {
			outputNodes.get(0).calculateOutput();
		}
		return outputNodes.get(0).getOutput() > 0.5 ? 1:0;
	}
	/*
	 * @param Instance inst
	 * @param int i
	 * standardize the numerical value for input
	 * */
	private double standardizeInput(Instance inst, int i) {
		AttributeStats attributeStat = trainingSet.attributeStats(i);
		double mean = attributeStat.numericStats.mean;
		double std = attributeStat.numericStats.stdDev;
		double inputDouble = (inst.value(i) - mean) / std;
		return inputDouble;
	}
	/*
	 * back-propagation with the neural network
	 * 
	 **/
	public void backPropLearning (Instances trainingSet) {
		int loop = this.maxEpoch;
		while (loop>=0) {
			Instances trainSet = new Instances(trainingSet);
			Random generator = new Random(loop);
			trainSet.randomize(generator);;
			loop--;
			for (Instance inst : trainingSet) {
				double[] inputDouble = inst.toDoubleArray();
				//initialize the input nodes
				for(int i = 0 ; i < inputDouble.length; i++) {
					if(inst.attribute(i).isNominal()) {
						List<Node> inputNodesList = inputNodes.get(i);
						for(Node n : inputNodesList){
							n.setInput(0.0);
						}
						inputNodesList.get((int) inputDouble[i]).setInput(1.0);
					}else if(inst.attribute(i).isNumeric()) {
						double in = standardizeInput(inst, i);
						List<Node> inputNodeList = inputNodes.get(i);
						inputNodeList.get(0).setInput(in);
					}
				}
				if(withHiddenNode){
					for(int i = 0; i < hiddenNodes.size(); i++) {
						hiddenNodes.get(i).calculateOutput();
					}
					outputNodes.get(0).calculateOutput();
					
				}else{
					outputNodes.get(0).calculateOutput();
				}
				
				double delta = inst.classValue() - outputNodes.get(0).getSum();
				
				for (int i = 0; i< input.size(); i++) 
				{
					inputNodes.get(i).setInput(input.get(i));
				}
				for (Node hiddenNode : hiddenNodes) 
				{
					hiddenNode.calculateOutput();
				}
				for (Node outputNode : outputNodes) 
				{
					outputNode.calculateOutput();
				}
				double [] output = new double[outputNodes.size()];
				for (int i =0; i < outputNodes.size(); i++) 
				{
					output[i] = outputNodes.get(i).getOutput();
				}
				double [] deltaOutput = new double[outputNodes.size()];
				for (int i =0; i< outputNodes.size(); i++) 
				{
					double x = outputNodes.get(i).getSum();
					deltaOutput[i] = (x <= 0? 0:1)*inst.classValues.get(i)-output[i];
				}

				double [] deltaHidden = new double[hiddenNodes.size()];
				for (int i=0 ; i < hiddenNodes.size(); i++) 
				{
					Node hiddenNode = hiddenNodes.get(i);
					double wmd=0;
					for (int j=0; j<outputNodes.size(); j++) 
					{
						Node outputNode = outputNodes.get(j);
						List<NodeWeightPair> pairs = outputNode.parents;
						for (NodeWeightPair np : pairs) 
						{
							if (np.node.equals(hiddenNode)) 
							{
								double temp = np.weight;
								wmd +=temp*deltaOutput[j];
							}
						}
					}
					double ini = hiddenNode.getSum();
					deltaHidden [i] = (ini <=0 ? 0:1) *wmd;
				}
				int j=0;
				for (Node outputNode : outputNodes) 
				{
					List<NodeWeightPair> pairs =  outputNode.parents;
					for (NodeWeightPair pair : pairs) 
					{
						pair.weight += learningRate * outputNode.getSum() * deltaOutput[j];//2. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!getSum or getOutput ?????
					}
					j++;
				}
				int i=0;
				for (Node hiddenNode : hiddenNodes) 
				{
					List<NodeWeightPair> pairs =  hiddenNode.parents;
					if (pairs!=null) 
					{ 
						for (NodeWeightPair pair : pairs) 
						{
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
