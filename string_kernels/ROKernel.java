/*
 *    ROKernel.java
 *
 *    Implementation of a Ratcliff-Obershelp kernel for Weka
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
import java.util.Stack;
import java.util.Vector;

public class ROKernel extends Kernel{

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
	public ROKernel(){
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
		// RO of two identical strings is 1, no need to cache
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

		// if either string is empty, make the distance small but not 0
		if (s1.length() == 0 || s2.length() == 0) {
			result = .001;
		} else {
			result = ratcliff_obershelp(s1, s2);
		}
		
		// store result in cache
		if (key != -1){
			m_storage[location] = result;
			m_keys[location] = (key + 1);
		}
		return result;
	}
	
	/**
	 * Computes the Ratcliff-Obershelp string-matching algorithm
	 * @param a
	 * @param b
	 * @return the "similarity" between the two strings
	 */
	public static double ratcliff_obershelp(String a, String b){
		int[][] lengths = new int[a.length()+1][b.length()+1];	 
		int longest_substring_len = 0;
		int[] longest_substring_coords = {0,0};
		for (int i = 0; i < a.length(); i++) {
			for (int j = 0; j < b.length(); j++) {
				if (a.charAt(i) == b.charAt(j)) {
					lengths[i+1][j+1] = lengths[i][j] + 1;
					if (lengths[i+1][j+1] > longest_substring_len){
						longest_substring_len = lengths[i+1][j+1];
						longest_substring_coords[0] = i+1;
						longest_substring_coords[1] = j+1;
					}
				} else {
					lengths[i+1][j+1] = 0;
				}
			}
		}
			
		double total_substring_len = longest_substring_len;
		
		Stack<int[][]> areas_to_search = new Stack<int[][]>();
		//left side of longest substring
		int[][] left = {new int[]{1, longest_substring_coords[0] - longest_substring_len}, 
				new int[]{1, longest_substring_coords[1] - longest_substring_len}};
		areas_to_search.push(left);
		//right side of longest substring
		int[][] right = {new int[]{longest_substring_coords[0]+1, a.length()}, 
				new int[]{longest_substring_coords[1]+1, b.length()}};
		areas_to_search.push(right);

		int largest_local_substring = 0;
		int[] best_local_coords = {0,0};
		int[][] nextArea;
		while(!areas_to_search.isEmpty()){
			nextArea = areas_to_search.pop();
			if(nextArea[0][0] > nextArea[0][1] || nextArea[1][0] > nextArea[1][1]){
				continue;
			}
			for(int i = nextArea[0][0]; i <= nextArea[0][1]; i++){
				for(int j = nextArea[1][0]; j <= nextArea[1][1]; j++){
					if(lengths[i][j] > largest_local_substring){
						largest_local_substring = lengths[i][j];
						best_local_coords[0] = i;
						best_local_coords[1] = j;
					}
				}
			}

			if(largest_local_substring > 0){
				// local left/right
				left[0] = new int[]{nextArea[0][0], best_local_coords[0] - largest_local_substring};
				left[1] = new int[]{nextArea[1][0], best_local_coords[1] - largest_local_substring};
				right[0] = new int[]{best_local_coords[0] + 1, nextArea[0][1]};
				right[1] = new int[]{best_local_coords[1] + 1, nextArea[1][1]};    
				
				areas_to_search.push(left);
				areas_to_search.push(right);
			}
			
			total_substring_len += largest_local_substring;
			largest_local_substring = 0;
		}
		return total_substring_len * 2 / (a.length() + b.length());
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
		return "Ratcliff-Obershelp kernel function";
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
