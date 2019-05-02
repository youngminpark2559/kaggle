# ================================================================================
Col-1: patient ID
Col-2: responder status ("1": improved to better, "0" otherwise)
Col-3: DNA sequence to create protease (enzyme protein) (if available)
Col-4: DNA sequence to create reverse transciptase (enzyme protein) (if available)
Col-5: "viral load severity" at the beginning of therapy (log-10 units)
Col-6: CD4 (cluster structures on the surface of immune cells such as T helper cells, monocytes, macrophages) count at the beginning of therapy

# ================================================================================
* Responder: indicates whether the patient improved after 16 weeks of therapy.

* Improvement is defined as a 100-fold decrease in the "HIV-1 viral load".

* There's a brief description of 
- Protease nucleotide sequence
- Reverse Transciptase nucleotide sequence
- viral load
- CD4 count
on the background page. 

# ================================================================================
training_data.csv: training dataset to calibrate your model. 

test_data.csv: test datset to generate submissions
