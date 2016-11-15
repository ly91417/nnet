package com.yingli.hw2;
/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */
import weka.core.*;
import java.util.*;
public class NNImpl{
	public boolean withHiddenNode = true;
	public ArrayList<Node> mInputNodeList = new ArrayList<>();
	//list of the input layer nodes
	public ArrayList<ArrayList<Node>> inputNodes = new ArrayList<>();
	//list of the hidden layer nodes
	public ArrayList<Node> hiddenNodes = new ArrayList<>();
	//list of the output layer nodes
	public ArrayList<Node> outputNodes = new ArrayList<>();
	//the training set
	public Instances trainingSet = null;
	//variable to store the learning rate
	private double learningRate = 1.0; 
	//variable to store the maximum number of epochs
	private int maxEpoch=1; 
	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */
	public NNImpl(Instances trainingSet, double learningRate, int hiddenNodeCount, int maxEpoch) {
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		if(hiddenNodeCount == 0 ) {
			withHiddenNode = false;
		}
		initializeNetworkWithRandomWeights(trainingSet, hiddenNodeCount);
	}
	/**
	 * initialize the neural network by using randomized weights
	 * @param Instances trainingSet
	 * @param int hiddenNodeCount
	 */
	private void initializeNetworkWithRandomWeights(Instances trainingSet, int hiddenNodeCount) {
		initializeInputLayer(trainingSet);
		initializeHiddenLayer(hiddenNodeCount);
		initializeOutputLayer(hiddenNodeCount);
	}
	/**
	 * initialize the output layer
	 * 
	 * */
	private void initializeOutputLayer(int hiddenNodeCount) {
		//Output node layer
		Node node=new Node(4);
		Random generator = new Random(hiddenNodeCount);
		if(withHiddenNode) {
			//Connecting output layer nodes with hidden layer nodes
			//j stand for index of hidden node
			for(int j = 0;j < hiddenNodes.size(); j++) {
				int number = generator.nextInt(201)-100;
				double initWt = number / 100.0;
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), initWt);
				node.parents.add(nwp);
			}	
		}
		//connecting output layer directly with input layers
		else {
			for(Node input :  this.mInputNodeList) {
				int number = generator.nextInt(201)-100;
				double initWt = number / 100.0;
				NodeWeightPair nwp = new NodeWeightPair(input, initWt);
				node.parents.add(nwp);
			}
		}
		outputNodes.add(node);
	}
	/**
	 * initialize the input layer
	 * */
	private void initializeInputLayer(Instances trainingSet) {
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
			//add the list to of the input nodes for certain feature to the list of lists
			inputNodes.add(inputNodeList);
		}
		//bias to hidden
		Node biasToHidden = new Node(1);
		ArrayList<Node> biasToHiddenList = new ArrayList<Node>();
		biasToHiddenList.add(biasToHidden);
		mInputNodeList.add(biasToHidden);
		inputNodes.add(biasToHiddenList);
	}
	/**
	 * initialize the hidden layer
	 * */
	private void initializeHiddenLayer(int hiddenNodeCount) {
		if(withHiddenNode) {
			//hidden layer nodes
			assert(hiddenNodeCount != 0):"hiddenNodeCount equals 0 and initialize with hidden layer";
			hiddenNodes = new ArrayList<Node> (hiddenNodeCount + 1);
			//i stand for index of hidden node
			for(int i = 0; i < hiddenNodeCount; i++) {
				//hidden node
				Node node = new Node(2);
				Random generator = new Random(hiddenNodeCount);
				//for each attribute
				for(int j = 0; j < inputNodes.size(); j++) {	
					int number = generator.nextInt(201) - 100;
					double initWt = number / 100.0;
					//for each value of the attribute
					for(Node n : inputNodes.get(i)) {
						NodeWeightPair nwp = new NodeWeightPair(n,initWt);
						node.parents.add(nwp);
					}
				}
				hiddenNodes.add(node);
			}
			//bias from hidden to output
			//type 3 is the bias node from hidden to output
			Node biasToOutput=new Node(3);
			hiddenNodes.add(biasToOutput);
		}
	}
	/**
	 *return 0 and 1 standing for the first and last class label
	 */
	public int classify(Instance inst)
	{
		//set up all the input values
		for(int i = 0 ; i < trainingSet.numAttributes() - 1; i++ ) {	
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
	 **/
	private void backPropLearning (Instances trainingSet) {
		int loop = this.maxEpoch;
		double[] error = new double[loop];
		if(withHiddenNode){
			while (loop>=0) {
				Instances trainSet = new Instances(trainingSet);
				Random generator = new Random(loop);
				trainSet.randomize(generator);
				
				for (Instance inst : trainingSet) {
					double[] inputDouble = inst.toDoubleArray();
					//initialize the input nodes
					setInputValues(inst, inputDouble);
					for(int i = 0; i < hiddenNodes.size(); i++) {
						hiddenNodes.get(i).calculateOutput();
					}
					outputNodes.get(0).calculateOutput();
					updateWeights(inst);
					error[loop-1] += cross_entropy(inst);
				}
				System.out.println("Epoch " + loop + " " + error[loop-1]);
				loop--;
			}	
		}else{
			while (loop>=0) {
				Instances trainSet = new Instances(trainingSet);
				Random generator = new Random(loop);
				trainSet.randomize(generator);
				
				for (Instance inst : trainingSet) {
					double[] inputDouble = inst.toDoubleArray();
					//initialize the input nodes
					setInputValues(inst, inputDouble);
					outputNodes.get(0).calculateOutput();
					updateWeights(inst);	
					error[loop] += cross_entropy(inst);
				}
				System.out.println("Epoch " + loop+1 + " " + error[loop-1]);
				loop--;
			}	
		}
	}
	/**
	 * return the corss_entropy of the instance
	 * */
	private double cross_entropy(Instance inst) {
		return (-inst.classValue()) * Math.log(outputNodes.get(0).getOutput()) 
				- (-inst.classValue()) * Math.log(1 - outputNodes.get(0).getOutput());
	}
	/*
	 * 
	 *set input values for the input nodes
	 */
	private void setInputValues(Instance inst, double[] inputDouble) {
		for(int i = 0 ; i < inputDouble.length-1; i++) {
			if(inst.attribute(i).isNominal()) {
				List<Node> inputNodesList = inputNodes.get(i);
				for(Node n : inputNodesList){
					n.setInput(0.0);
				}
				int index = (int)inputDouble[i];
				System.out.println(inputNodesList.size() + " " + index );
				inputNodesList.get(index).setInput(1.0);
			}else if(inst.attribute(i).isNumeric()) {
				double in = standardizeInput(inst, i);
				List<Node> inputNodeList = inputNodes.get(i);
				inputNodeList.get(0).setInput(in);
			}
		}
		
	}
	private void updateWeights(Instance inst) {
		double output = outputNodes.get(0).getOutput();
		//deltak = ok(1-ok)(tk-ok)
		//refer to McGrawHill Machine_learning Mitchell p98 formula T4.3
		double deltaK = (inst.classValue() - output) * output * (1 - output);
		for(NodeWeightPair hnp : outputNodes.get(0).parents ) {
			Node hiddenN = hnp.node;
			if(hiddenN.type !=3) {
				double hiddenWt = hnp.weight;
				double hiddenOutput = hiddenN.getOutput();
				//refer to McGrawHill Machine_learning Mitchell p98 formula T4.4
				double deltaH = hiddenOutput * (1- hiddenOutput) * (hiddenWt * deltaK);
				//refer to McGrawHill Machine_learning Mitchell p98 formula T4.5
				double deltaWtji = learningRate * deltaK * hiddenN.getOutput();
				hiddenWt += deltaWtji;
				for(NodeWeightPair inp: hiddenN.parents) {
					Node inputN = inp.node;
					//refer to McGrawHill Machine_learning Mitchell book p98 formula T4.5
					double deltaWthj = learningRate * deltaH * inputN.getOutput();
					inp.weight += deltaWthj;
				}
			}else {
				//TODO need to implement the hiddenNode weight update and need to implement 
				//weight update without hidden nodes
				double hiddenWt = hnp.weight;
				double hiddenOutput = hiddenN.getOutput();
				//refer to McGrawHill Machine_learning Mitchell p98 formula T4.4
				double deltaH = hiddenOutput * (1- hiddenOutput) * (hiddenWt * deltaK);
				//refer to McGrawHill Machine_learning Mitchell p98 formula T4.5
				double deltaWtji = learningRate * deltaK * hiddenN.getOutput();
				hiddenWt += deltaWtji;
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
		backPropLearning(this.trainingSet);
	}
}
