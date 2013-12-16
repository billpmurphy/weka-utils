/*
 *    LevenshteinKernel.java
 *
 *    Implementation of an edit-distance (Levenshtein distance) string kernel for Weka
 *
 */

import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import java.util.Enumeration;
import java.util.Vector;

public class LevenshteinKernel extends Kernel{

	private int m_kernelEvals;
	private int m_strAttr;
	private double[] m_storage;
	private long[] m_keys;
	private int m_numInsts;
	private int m_cacheSize = 250007;
	private int m_internalCacheSize = 200003;

	/**
	 * default constructor
	 */
	public LevenshteinKernel(){
		super();
	}

	/**
	 * Initializes class variables and the cache.
	 * @param data the dataset to use
	 */
	protected void initVars(Instances data) {
		super.initVars(data);
		m_kernelEvals = 0;

		// find the first string attribute
		m_strAttr = -1;
		for (int i = 0; i < data.numAttributes(); i++) {
			if (i == data.classIndex())
				continue;
			if (data.attribute(i).type() == Attribute.STRING) {
				m_strAttr = i;
				break;
			}
		}
		m_numInsts = m_data.numInstances();
		m_storage = new double[m_cacheSize];
		m_keys = new long[m_cacheSize];
	}

	/**
	 * Computes the result of the kernel function for two instances.
	 * If id1 == -1, eval use inst1 instead of an instance in the dataset.
	 * @param id1 the index of the first instance in the dataset
	 * @param id2 the index of the second instance in the dataset
	 * @param inst1 the instance corresponding to id1 (used if id1 == -1)
	 * @return the result of the kernel function
	 * @throws Exception if something goes wrong
	 */
	public double eval(int id1, int id2, Instance inst1) throws Exception {
		// kernel result of two identical strings is 1, no need to cache
		if(id1 == id2 && id1 == -1) {
			return 1;
		}

		double result = 0;
		long key = -1;
		int location = -1;

		// we can access the cache if we know the indexes
		if ((id1 >= 0) && (m_keys != null)) {
			if (id1 > id2) {
				key = (long)id1 * m_numInsts + id2;
			} else {
				key = (long)id2 * m_numInsts + id1;
			}
			if (key < 0) {
				throw new Exception("Cache overflow detected!");
			}
			location = (int)(key % m_keys.length);
			if (m_keys[location] == (key + 1)) {
				return m_storage[location];
			}
		}

		// did not use the cache; compute the kernel
		m_kernelEvals++;
		String s1 = inst1.stringValue(m_strAttr);
		String s2 = m_data.instance(id2).stringValue(m_strAttr);

        // find the edit distance and take the inverse
		result = 1.0/(.001 + levenshtein(s1, s2));

		// store result in cache
		if (key != -1){
			m_storage[location] = result;
			m_keys[location] = (key + 1);
		}
		return result;
	}

	/**
	 * Computes the edit distance (Levenshtein distance) between two strings
	 * @param str1 the first string
	 * @param str2 the second string
	 * @return the edit distance -- this should be normalized by dividing by the length of the two strings
	 */
	public double levenshtein(String str1, String str2){
		int[][] distance = new int[str1.length() + 1][str2.length() + 1];
		for (int i = 0; i <= str1.length(); i++) {
			distance[i][0] = i;
		}
		for (int j = 1; j <= str2.length(); j++) {
			distance[0][j] = j;
		}
		for (int i = 1; i <= str1.length(); i++) {
			for (int j = 1; j <= str2.length(); j++) {
				distance[i][j] = Math.min(
						Math.min(distance[i - 1][j] + 1, distance[i][j - 1] + 1), 
						distance[i - 1][j - 1] + ((str1.charAt(i - 1) == str2.charAt(j - 1)) ? 0 : 1));
			}
		}
		return distance[str1.length()][str2.length()];    
	}


	/**
	 * Returns the number of kernel evaluation performed.
	 * @return the number of kernel evaluation performed.
	 */
	public int numEvals() {
		return m_kernelEvals;
	}

	/**
	 * Returns a string describing the kernel
	 * @return a description suitable for displaying in the gui
	 */
	public String globalInfo() {
		return "Levenshtein string kernel function";
	}

	/**
	 * Clear the cache
	 */
	public void clean() {
		m_storage = null;
		m_keys = null;
	}

	/**
	 * Not supported by this kernel
	 */
	public int numCacheHits() {
		return -1;
	}

}

