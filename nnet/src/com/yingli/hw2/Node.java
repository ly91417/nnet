package com.yingli.hw2;
import java.util.*;

public class Node {
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	//ArrayList that will contain the parents (including the bias node) with weights if applicable	
	public ArrayList<NodeWeightPair> parents=null; 
	private Double inputValue=0.0;
	private Double outputValue=0.0;
	private Double sum=0.0; // sum of wi*xi
	//Create a node with a specific type
		public Node(int type)
		{
			if(type>4 || type<0)
			{
				System.out.println("Incorrect value for node type");
				System.exit(1);
				
			}
			else
			{
				this.type=type;
			}
			
			if (type==2 || type==4)
			{
				parents=new ArrayList<NodeWeightPair>();
			}
		}
		
		//For an input node sets the input value which will be the value of a particular attribute
		public void setInput(Double inputValue)
		{
			if(type==0)//If input node
			{
				this.inputValue=inputValue;
			}
		}
		
		/**
		 * Calculate the output of a ReLU node.
		 * You can assume that outputs of the parent nodes have already been calculated
		 * You can get this value by using getOutput()
		 * @param train: the training set
		 */
		public void calculateOutput()
		{
			
			if(type==2 || type==4)//Not an input or bias node
			{
				// TODO: add code here
				for (NodeWeightPair np : parents) {
					// if (np.node.type == 1 ||np.node.type == 3 ){
					// 	sum += (double) np.weight* np.node.getOutput();
					// }else
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
		public double getSum() {
			return sum;
		}
		
		//Gets the output value
		public double getOutput()
		{
			
			if(type==0)//Input node
			{
				return inputValue;
			}
			else if(type==1 || type==3)//Bias node
			{
				return 1.00;
			}
			else
			{
				return outputValue;
			}
			
		}
}
