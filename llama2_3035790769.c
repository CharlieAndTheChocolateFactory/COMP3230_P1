,/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME:llama2_3035790769.c
* NAME:Wong Chor Sing
* UID :3035790769
* Development Platform:VScode on mac
* Remark: I fufilled most of the requirements
* How to compile: (gcc -o llama2_3035790769 llama2_3035790769.c utilities.c -O2 -pthread -lm)

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o llama2_3035790769 llama2_3035790769.c utilities.c -O2 -pthread -lm


Then Run with:
$ ./llama2_3035790769 <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each row, so we can use Multi-Threading for acceleration.
 * 
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 * 
 * A sequential version is provided in seq.c, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>

// Global Variables
struct rusage main_usage;  // get usage for main thread
pthread_t* thread;
pthread_mutex_t lock;
pthread_cond_t cond;
pthread_cond_t condStart;
pthread_cond_t condEnd;
int* threadRunning;
int NUM_THREADS;
int task = 0;
struct matric_arg {
  float* out;
  float* vec;
  float* mat;
  int col;
  int row;
};

struct matric_arg* matric_arg_1;
void* thr_func(void* arg);

int init_mat_vec_mul(int thr_count) {  // main thread for initialization
  thread = malloc(sizeof(pthread_t) * thr_count); 
  matric_arg_1 = malloc(sizeof(matric_arg_1[0]) * thr_count);
  threadRunning = malloc(sizeof(int) * thr_count);
  NUM_THREADS = thr_count;
  pthread_mutex_init(&lock, NULL);
  pthread_cond_init(&condStart, NULL);
  pthread_cond_init(&condEnd, NULL);
  for (int i = 0; i < NUM_THREADS; i++) {
    int* tid = malloc(sizeof(int));
    *tid = i; //Let threads identify themselves, i.e., each thread knows it is the i-th threads
    pthread_create(thread + i, NULL, thr_func, tid); //Create n threads
  }
  return 0;
}

void mat_vec_mul(float* out, float* vec, float* mat, int col,
                 int row) {  // main thread for multipication
  for (int i = 0; i < NUM_THREADS; i++) { //Assign new parameters (out, vec, mat, col, row) to threads
    float val = 0.0f;
    (matric_arg_1 + i)->out = out;
    (matric_arg_1 + i)->vec = vec;
    (matric_arg_1 + i)->mat = mat;
    (matric_arg_1 + i)->col = col;
    (matric_arg_1 + i)->row = row;
    
  }
  pthread_mutex_lock(&lock);
  for (int i = 0; i < NUM_THREADS; i++) {
    threadRunning[i] = 1;
  }
  pthread_cond_broadcast(&condStart); //Wake up threads to do calculation
  pthread_mutex_unlock(&lock);

  pthread_mutex_lock(&lock);
  for (int i = 0; i < NUM_THREADS; i++) {
    while (threadRunning[i] != 0) {
      pthread_cond_wait(&condEnd, &lock); //Main thread wait until all threads finished task
    }
  }
  pthread_mutex_unlock(&lock);
}

int close_mat_vec_mul() {  // main thread for terminaiton/clearing
  task = 1;
  for (int i = 0; i < NUM_THREADS; i++) {
    threadRunning[i] = 1;
  }
  pthread_cond_broadcast(&condStart); //Wake up threads to collect the system usage (of themselves) and terminates
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread[i], NULL); //Wait until all threads to exit and collect system usage of threads
  }
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  struct timeval usertime = usage.ru_utime; //Collect system usage of main thread, and display both usage of each thread and main thread
  struct timeval systime = usage.ru_stime;

  float Utime = usertime.tv_sec + (float)usertime.tv_usec / 1000000;
  float Stime = systime.tv_sec + (float)systime.tv_usec / 1000000;

  printf("main thread - user: %0.4f s, system: %0.4f s\n", Utime, Stime);
  //Clear all resources related with multi-threading, and return
  free(thread);
  free(matric_arg_1);
  pthread_mutex_destroy(&lock); 
  pthread_cond_destroy(&condStart);
  pthread_cond_destroy(&condEnd);

 
}

void* thr_func(void* arg) {  // child thread for multipication and return data
  int index = *(int*)arg; 

  while (1) {
    pthread_mutex_lock(&lock);
    while (threadRunning[index] == 0) {
      pthread_cond_wait(&condStart, &lock); //Let the created threads fall asleep immediately
    } //Can be woke up by main thread to work on assigned tasks
    pthread_mutex_unlock(&lock);

    if (task == 1) {
      break;
    }

    struct matric_arg matrixStruct = matric_arg_1[index];
    int rowStart = index * matrixStruct.row / NUM_THREADS;
    int rowEnd = (index + 1) * matrixStruct.row / NUM_THREADS;
    for (int i = rowStart; i < rowEnd; i++) {
      float val = 0.0f;
      for (int j = 0; j < matrixStruct.col; j++) {
        val += matrixStruct.mat[i * matrixStruct.col + j] *
               matrixStruct.vec[j];  // mat[i * col + j] := mat[i][j] }
      }
      matrixStruct.out[i] = val;
    }
 

    pthread_mutex_lock(&lock);
    threadRunning[index] = 0;
    pthread_cond_broadcast(&condEnd);//After finishing the task, inform main thread
    pthread_mutex_unlock(&lock);
  }
  struct rusage usage;
  getrusage(RUSAGE_THREAD, &usage);
  struct timeval usertime = usage.ru_utime;//Being able to collect the system usage (of itself) and terminate
  struct timeval systime = usage.ru_stime;

  float Utime = usertime.tv_sec + (float)usertime.tv_usec / 1000000;
  float Stime = systime.tv_sec + (float)systime.tv_usec / 1000000;

  printf("Thread %d has completed - user: %0.4f s, system: %0.4f s\n", index, Utime,
         Stime);
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %0.4f s, achieved tok/s: %0.4f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}