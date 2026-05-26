# Het-CPFLoRA
  
## Running

Example usage:
```bash
CUDA_VISIBLE_DEVICES=0 python main_my_het.py --global_model 'meta-llama/Llama-3.2-1B'      --data_path  "../data/dataset2"       --output_dir  '../lora-1b-8-het-data2/'      --num_communication_rounds 10       --num_clients  8       --prompt_template_name 'alpaca_short'       --client_selection_frac 1       --local_model False
```

## Inference 

```bash
python kunhuo2.py --r $r_s$ --allr $r$ --file lora-1b-8-het-data2 --local 0  --data data2
      
```
