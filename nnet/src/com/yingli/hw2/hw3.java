package com.yingli.hw2;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
public class hw3 {
	public static void main(String[] args) {
		if (args.length	!= 5)
		{
			System.out.println("usage: nnet l h e <train-set-file> <test-set-file>");
			pukeAndDie();
		}
		DataSource train_set_file;
		try {
			train_set_file = new DataSource(args[3]);
			DataSource test_set_file = new DataSource(args[4]);
			Instances train_set = train_set_file.getDataSet();
			Instances test_set = test_set_file.getDataSet();
			if (train_set.classIndex() == -1) {
				train_set.setClassIndex(train_set.numAttributes() - 1);	
			}
			if (test_set.classIndex() == -1) {
				test_set.setClassIndex(test_set.numAttributes() - 1);	
			}
			double l = Double.valueOf(args[0]);
			int h = Integer.valueOf(args[1]);
			int epoch = Integer.valueOf(args[2]);
			NNImpl network = new NNImpl(train_set,l,h,epoch);
			network.train();
			for(Instance in : test_set) {
				int classLabel = network.classify(in);
				System.out.println("the instance class value is " + in.classValue() 
				+ "and the output is " + classLabel );
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	private static void pukeAndDie() {
		System.out.println("puke and die");
		System.exit(-1);
	}
}
