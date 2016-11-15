package com.yingli.hw2;
import java.util.*;
public class Node {
	public int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	//ArrayList that will contain the parents (including the bias node) with weights if applicable	
	public ArrayList<NodeWeightPair> parents=null; 
	private double inputValue = 0.0;
	private double outputValue = 0.0;
	private double sum = 0.0; // sum of wi*xi
	//Create a node with a specific type
		public Node(int type) {
			if(type>4 || type<0) {
				System.out.println("Incorrect value for node type");
				System.exit(1);
			}
			else {
				this.type=type;
			}
			
			if (type==2 || type==4) {
				parents=new ArrayList<NodeWeightPair>();
			}
		}
		//For an input node sets the input value which will be the value of a particular attribute
		public void setInput(double inputValue) {
			if(type==0) {
				this.inputValue=inputValue;
			}
		}
		/**
		 * Calculate the output of a ReLU node.
		 * You can assume that outputs of the parent nodes have already been calculated
		 * You can get this value by using getOutput()
		 * @param train: the training set
		 */
		public void calculateOutput() {
			
			if(type==2 || type==4) {
				// TODO: add code here
				for (NodeWeightPair np : parents) { 
					sum += (double) np.weight * np.node.getOutput();
				}
				this.outputValue = getSigmoidalOutput(sum); 
			}
		}
		/**
		 * getSigmoidalOutput
		 */
		private double getSigmoidalOutput(double value) {
			return (1.0 / (1.0 + Math.pow(Math.E, -value)));
		}
		/**
		 * return sum of the num
		 * 
		 * */
		public double getSum() {
			return sum;
		}
		/**
		 * get output for the node
		 * */
		public double getOutput() {
			//for input nodes
			if(type==0) {
				return inputValue;
			}
			//for bias nodes
			else if(type==1 || type==3) {
				return 1.00;
			}
			//for output nodes
			else {
				return outputValue;
			}
		}
}
