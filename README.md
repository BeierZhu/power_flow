## Calculating power flow with XB, XX, XX, BB and Coupled Constant Jacobian methods
---
	The cases {14, 39, 57, 118, 2383wp} used in this experiment are exported from Matpower.
	
###Usage
**example**

~~~
python runpf.py -m 1 --case_number --ksai 1e-5 --max_iter 500 --verbose 0 --scale 1
~~~

#### -m: iterator   
	0: constant Jacobian  
	1: 'BB'  
	2: 'BX'  
	3: 'XB'  
	4: 'XB_r' count resistances of branch in B prime  
	5: 'XB_ground' count effect of grounding branch
	default=2
#### --case_number:  
	14, 39, 57, 118, 2383wp

#### --ksai:
	converage criterion 
	default=1e-6

#### --max_iter:
	default=1e3

#### --verbose:
	0: less information
	1: information during each iteration
	2: more information
	default:0
	
#### --scale
	r/x scale:
	if scale > 1, r*scale/x
	else r/(x*scale)
	default:1

### Run experiment in my report
results found in folder experiement_log
#### experiment 3.1
	./accuracy_test.sh
#### experiment 3.2
	./iteration_test.sh+case_number
exemple:  
	./iteration 14
#### experiment 3.3.1
	./XB_count_r_test.sh+case_number
#### experiment 3.3.2
	./XB_count_ground_test.sh+case_number
#### experiemnt 3.4
	./XB_BX_comparison.sh+case_number
	./XB_BX_comparison2.sh+case_number
	