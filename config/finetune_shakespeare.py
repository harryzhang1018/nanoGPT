# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

out_dir = 'out-shakespeare-char-66396'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'allic_in_wdld'
init_from = 'resume'
gradient_accumulation_steps = 1
batch_size = 12
# context of up to 256 previous characters
block_size = 64


# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 396
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 3000
lr_decay_iters = 3000 # make equal to max_iters usually


warmup_iters = 100 # not super necessary potentially


