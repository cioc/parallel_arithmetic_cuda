#include <stdio.h>
#include <string.h>
#include <math.h>

/*
	TODO project: reciprocal 
*/

/*
	TODO project: negative number human readable
*/

//KEY CONSTANTS

//THIS IS THE STARTING SIZE FOR THE TWO TEMP BUFFERS 
//THE LARGER YOU EXPECT YOUR NUMBERS TO GROW, INCREASES THE SIZE OF THIS VALUE TO INCREASE PERFORMANCE
const unsigned int ORIGINAL_TEMP_BUFFER_SIZE = 1024;
//THESE TWO WILL SERVE AS OUR TEMPORARY BUFFERS IN OUR COMPUTATIONS
int *temp_buffer1;
int *temp_buffer2;
int *temp_buffer3;
unsigned int temp_buffer_size;

//NORMALIZATION KEY VALUES
const unsigned int BITS_PER_DIGIT = 32;
const unsigned int NORMALIZATION_EXPANSION = (unsigned int)ceil((BITS_PER_DIGIT * log(2.0)) / (log(10.0)));

//KEY PROCESSING CONSTANTS
//THIS IS THE DEVICE NUMBER THAT WE WILL DO OUR CALCULATIONS ON
const int DEVICE_NUM = 0;
int MAX_THREADS_PER_BLOCK;

//THE FOLLOWING IS HELPFUL INPUT CODE
//BCD - binary coded decimal
//A BCD IS THE DATA STRUCTURE THAT WE WILL USE TO REPRESENT OUR LARGE NUMBERS

//decpos IS THE POSITION OF THE DECIMAL IN THE NUMBER
//length IS THE NUMBER OF DIGITS IN THE NUMBER
//values IS AN ARRAY OF THE DIGITS
//gpuP IS THE POINTER TO THE DIGITS THAT HAVE BEEN COPIED TO THE GPU'S MEMORY
typedef struct bcd {
	unsigned int decpos;
	unsigned int length; 
	int *values; 
	int *gpuP;
} bdc;

//THIS TAKES A STRING REPRESENTATION OF OUR NUMBER, SUCH AS "123456544.23" AND LOADS IT INTO A BCD
void bcdFromString(char* input, bcd* output);
//THIS CREATES A BCD THAT CAN STORE A NUMBER WITH len DIGITS
bcd* createBcd(unsigned int len);
//THIS PRINTS A BCD OUT TO THE CONSOLE
void printBcd(bcd* input);
void printBcdNotNormal(bcd* input);
void zeroBcd(bcd* input);
void freeBcd(bcd* input);

//THE IMPLEMENTATION OF THESE THREE FUNCTIONS FOLLOWS:

void bcdFromString(char* input, bcd* output)
{
	unsigned int len = strlen(input);
	unsigned int lenstore = len;
	unsigned int x = 0;
	unsigned char decFound = 0;
	unsigned int negative = 0;
	for (x = 0; x < len; ++x)
	{
		char temp = input[x];
		switch (temp)
		{
			case '-':
				//lenstore -= 1;
				negative = 1;
			break;
			case '0':
				if (decFound > 0)
				{
					output->values[x - 1] = 0;
				}
				else
				{
					output->values[x] = 0;
				}
			break;
			case '1':
				if (decFound > 0)
				{
					output->values[x - 1] = 1;
				}
				else
				{
					output->values[x] = 1;
				}
			break;
			case '2':
				if (decFound > 0)
				{
					output->values[x - 1] = 2;
				}
				else
				{
					output->values[x] = 2;
				}
			break;
			case '3':
				if (decFound > 0)
				{
					output->values[x - 1] = 3;
				}
				else
				{
					output->values[x] = 3;
				}
			break;
			case '4':
				if (decFound > 0)
				{
					output->values[x - 1] = 4;
				}
				else
				{
					output->values[x] = 4;
				}
			break;
			case '5':
				if (decFound > 0)
				{
					output->values[x - 1] = 5;
				}
				else
				{
					output->values[x] = 5;
				}
			break;
			case '6':
				if (decFound > 0)
				{
					output->values[x - 1] = 6;
				}
				else
				{
					output->values[x] = 6;
				}
			break;
			case '7':
				if (decFound > 0)
				{
					output->values[x - 1] = 7;
				}
				else
				{
					output->values[x] = 7;
				}
			break;
			case '8':
				if (decFound > 0)
				{
					output->values[x - 1] = 8;
				}
				else
				{
					output->values[x] = 8;
				}
			break;
			case '9':
				if (decFound > 0)
				{
					output->values[x - 1] = 9;
				}
				else
				{
					output->values[x] = 9;
				}
			break;
			case '.':
				output->decpos = x;
				lenstore -= 1;
				decFound = 1;
			break;
		}
	}
	output->length = lenstore;
	if (negative == 1)
	{
		int i = 0;
		for(i = 0; i < lenstore; i++)
		{
			output->values[i] = output->values[i] * (-1);
		}
	}
	if (decFound == 0)
	{
		output->decpos = lenstore;
	}
}

bcd* createBcd(unsigned int len)
{
	bcd* output = (bcd *)malloc(sizeof(bcd));
	output->length = len;
	output->values = (int *)malloc(len * sizeof(int));
	return output;
}

void zeroBcd(bcd* input)
{
	int c = 0;
	for (c = 0; c < input->length; ++c)
	{
		*(input->values + c) = 0; 
	}
}

void printBcd(bcd* input)
{
	int i = 0;
	for(i = 0; i < input->length; i++)
	{
		if (i == input->decpos)
		{
			printf(".");
		}
		printf("%i", input->values[i]);
	}
	printf("\n");
}

void printBcdNotNormal(bcd* input)
{
	int i = 0;
	for(i = 0; i < input->length; i++)
	{
		if (i == input->decpos)
		{
			printf(".");
		}
		printf("%i", input->values[i]);
		printf("|");
	}
	printf("\n");
}

void freeBcd(bcd* input)
{
	cudaFree(input->gpuP);
	free(input->values);
	free(input);
}
//cudaFree

//GPU CODE

//THIS FUNCTION LOADS THE VALUES OF A BCD INTO TEH GPU'S MEMORY AND SETS THE gpuP OF THE BCD TO POINT TO THE GPU-STORED VALUES
void loadBcdIntoGPU(bcd* input);
//THIS COPIES BACK THE RESULTS FROM THE GPU TO THE BCD
void getCompResult(bcd* output);

void loadBcdIntoGPU(bcd* input)
{
	cudaMalloc(&input->gpuP, input->length * sizeof(int));
	cudaMemcpy(input->gpuP, input->values, input->length * sizeof(int), cudaMemcpyHostToDevice);
}
void getCompResult(bcd* output)
{
	cudaMemcpy(output->values,output->gpuP, output->length * sizeof(int), cudaMemcpyDeviceToHost);
}

//THE FOLLOWING IS ALL SETUP CODE

void cudaSetup();
void initTempBuffers();
void reallocTempBuffers(unsigned int size);
void freeTempBuffers();
void zeroTempBuffers();

//THIS IS THE MAIN SETUP FUNCTION.  CALL THIS EARLY ON IN MAIN.  BEFORE ANY ADDITIONS OR MULTIPLICATIONS ON BCD'S
//will call initTempBuffers
void cudaSetup()
{
	//LET'S FIGURE OUT THE MAXIMUM THREADS PER BLOCK
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DEVICE_NUM);
	MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
	initTempBuffers();
}


void initTempBuffers()
{
	cudaMalloc(&temp_buffer1, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));
	cudaMalloc(&temp_buffer2, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));
	cudaMalloc(&temp_buffer3, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));
	cudaMemset(temp_buffer1, 0, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));
	cudaMemset(temp_buffer2, 0, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));
	cudaMemset(temp_buffer3, 0, ORIGINAL_TEMP_BUFFER_SIZE * sizeof(int));	
	temp_buffer_size = ORIGINAL_TEMP_BUFFER_SIZE;
}

void reallocTempBuffers(unsigned int size)
{
	if (temp_buffer_size < size)
	{
		freeTempBuffers();
		cudaMalloc(&temp_buffer1, size * sizeof(int));
		cudaMalloc(&temp_buffer2, size * sizeof(int));
		cudaMalloc(&temp_buffer3, size * sizeof(int));
		cudaMemset(temp_buffer1, 0, size * sizeof(int));
		cudaMemset(temp_buffer2, 0, size * sizeof(int));
		cudaMemset(temp_buffer3, 0, size * sizeof(int));
		temp_buffer_size = size;
	}
}

void zeroTempBuffers()
{
	cudaMemset(temp_buffer1, 0, temp_buffer_size * sizeof(int));
	cudaMemset(temp_buffer2, 0, temp_buffer_size * sizeof(int));
	cudaMemset(temp_buffer3, 0, temp_buffer_size * sizeof(int));
}

void freeTempBuffers()
{
	cudaFree(temp_buffer1);
	cudaFree(temp_buffer2);
	cudaFree(temp_buffer3);
}

//MEMORY REQUIREMENT CALCULATION CODE

unsigned int memReqForAddition(bcd* num1, bcd* num2);
unsigned int memReqForMulitiplcation(bcd* num1, bcd* num2);

unsigned int memReqForAddition(bcd* num1, bcd* num2)
{
	unsigned int maxlen = 0; 
	if (num1->length > num2->length)
	{
		maxlen = num1->length;
	}
	else
	{
		maxlen = num2->length;
	}
	return (maxlen + NORMALIZATION_EXPANSION + 1);
}

unsigned int memReqForMulitiplcation(bcd* num1, bcd* num2)
{
	return num1->length + num2->length + (2 * NORMALIZATION_EXPANSION);
}

//DECIMAL POSITION CHANGE CODE

unsigned int decimalMovementAddition(bcd* num1, bcd* num2, unsigned int memReq);
unsigned int decimalMovementMultiplication(bcd* num1, bcd* num2, unsigned int memReq);

unsigned int decimalMovementAddition(bcd* num1, bcd* num2, unsigned int memReq)
{
	if (num1->length > num2->length)
	{
		return (memReq - num1->length) + num1->decpos;
	}
	else
	{
		return (memReq - num2->length) + num2->decpos;
	}
}

/*
	TODO START HERE
*/
unsigned int decimalMovementMultiplication(bcd* num1, bcd* num2, unsigned int memReq)
{
	return memReq - ((num1->length - num1->decpos) + (num2->length - num2->decpos));
}

//KERNELS

__global__ void addition(int *num1, int *num2, unsigned int num1Len, unsigned int num2Len, unsigned int num1offset, unsigned int num2offset, int *temp_buffer1, int *temp_buffer2, int *output, unsigned int memReq, unsigned int reps);
__global__ void normalize(int *num,unsigned int num1Len, int *result,unsigned int memReq);
__global__ void multiplication(int *num1, int *num2, unsigned int num1Len, unsigned int num2Len, unsigned int num1offset, unsigned int num2offset, int *temp_buffer1, int *temp_buffer2, int *temp_buffer3, int *output, unsigned int memReq, unsigned int reps);

/*
	TODO normalize: need to get this working for reps.  EG: numbers longer the 512 digits
*/
/*
	TODO normalize: need to make this work for negative numbers
*/

__global__ void normalize(int *num, unsigned int numLen, int *result,unsigned int memReq, unsigned int reps)
{
	int x = threadIdx.x;
	if (x < memReq)
	{
		if (reps == 1)
		{
			if (x >= memReq - numLen)
			{
				result[x] = num[x - (memReq - numLen)];
			}
		}
		else
		{
			//result[x] = num[x - (memReq - numLen)];
			int d = 0;
			for (d = 0; d < reps; ++d)
			{
				if (d == 0)
				{
					if (x >= memReq - numLen)
					{
						result[x + (512 * d)] = num[x + (512 * d) - (memReq - numLen)];
					}
				}
				else
				{
					if ((x + (512 * d) - (memReq - numLen)) < numLen)
					result[x + (512 * d)] = num[x + (512 * d) - (memReq - numLen)];
				}
			}
		}
	}
	
	__shared__ int carry;
	carry = 1;
	__syncthreads();
	while (carry)
	{
		if (reps == 1)
		{
			int c = 0;
			if (x < memReq)
			{
				c = result[x] / 10;
				result[x] %= 10;
			}
			__syncthreads();
			if (x < memReq && x != 0)
			{
				result[x - 1] += c;
			}
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				int c = 0;
				if ((x + (512 * d)) < memReq)
				{
					c = result[x + (512 * d)] / 10;
					result[x + (512 * d)] %= 10;
				}
				__syncthreads();
				if ((x + (512 * d)) < memReq && (x != 0))
				{
					result[x - 1 + (512 * d)] += c;
				}
			}
		}
		carry = 0;
		__syncthreads();
		if (x < memReq)
		{
			if (reps == 1)
			{
				if (abs(result[x]) > 9)
				{
					carry = 1;
				}
			}
			else
			{
				int d = 0;
				for(d = 0; d < reps; ++d)
				{
					if ((x + (512 * d)) < memReq && (abs(result[x + (512 * d)]) > 9))
					{
						carry = 1;
					}
				}
			}
		}
		__syncthreads();
	}
}

/*
	TODO addition kernel: need to add in normalization
*/
__global__ void addition(int *num1, int *num2, unsigned int num1Len, unsigned int num2Len, unsigned int num1offset, unsigned int num2offset, int *temp_buffer1, int *temp_buffer2, int *output, unsigned int memReq, unsigned int reps)
{
	int x = threadIdx.x;
	if (reps == 1)
	{
		if (x < memReq)
		{
			if (x >= memReq - num1Len)
			{
				temp_buffer1[x] = num1[x - (memReq - num1Len)];
			}
			if (x >= memReq - num2Len)
			{
				temp_buffer2[x] = num2[x - (memReq - num2Len)];
			}
		}
	}
	else
	{
		int d = 0;
		for (d = 0; d < reps; ++d)
		{
			if (d == 0)
			{
				if (x >= memReq - num1Len)
				{
					temp_buffer1[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
				}
				if (x >= memReq - num2Len)
				{
					temp_buffer2[x + (512 * d)] = num2[x + (512 * d) - (memReq - num2Len)];
				}
			}
			else
			{
				if ((x + (512 * d) - (memReq - num1Len)) < num1Len)
				{
					temp_buffer1[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
				}
				if ((x + (512 * d) - (memReq - num2Len)) < num2Len)
				{
					temp_buffer2[x + (512 * d)] = num2[x + (512 * d) - (memReq - num2Len)];
				}
			}
		}
	}
	//move everything to temp buffers
	__shared__ int carry;
	carry = 0;
	__syncthreads();
	if (reps == 1)
	{
		if (((unsigned int)(temp_buffer1[x] & ((unsigned int)3 << 30)) > 0) || ((unsigned int)(temp_buffer2[x] & ((unsigned int)3 << 30)) > 0))
		{
			carry = 1;
		}
	}
	else
	{
		int d = 0;
		for (d = 0; d < reps; ++d)
		{
			if ((x + (512 * d)) < memReq)
			{
				if (((unsigned int)(temp_buffer1[x + (512 * d)] & ((unsigned int)3 << 30)) > 0) || ((unsigned int)(temp_buffer2[x + (512 * d)] & ((unsigned int)3 << 30)) > 0))
				{
					carry = 1;
				}
			}
		}
	}
	__syncthreads();
	while (carry)
	{
		if (reps == 1)
		{
			int c1 = 0;
			int c2 = 0;
			if (x < memReq)
			{
				c1 = temp_buffer1[x] / 10;
				c2 = temp_buffer2[x] / 10;
				temp_buffer1[x] %= 10;
				temp_buffer2[x] %= 10;
			}
			__syncthreads();
			if (x < memReq && x != 0)
			{
				temp_buffer1[x - 1] += c1;
				temp_buffer2[x - 1] += c2;
			}
			carry = 0;
			__syncthreads();
			if (x < memReq)
			{
				if ((abs(temp_buffer1[x]) > 9) || (abs(temp_buffer2[x]) > 9))
				{
					carry = 1;
				}
			}
			__syncthreads();
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				int c1 = 0;
				int c2 = 0;
				if ((x + (512 * d)) < memReq)
				{
					c1 = temp_buffer1[x + (512 * d)] / 10;
					c2 = temp_buffer2[x + (512 * d)] / 10;
					temp_buffer1[x + (512 * d)] %= 10;
					temp_buffer2[x + (512 * d)] %= 10;
				}
				__syncthreads();
				if ((x + (512 * d)) < memReq && (x != 0))
				{
					temp_buffer1[x - 1 + (512 * d)] += c1;
					temp_buffer2[x - 1 + (512 * d)] += c2;
				}
			}
		}
		carry = 0;
		__syncthreads();
		if (x < memReq)
		{
			if (reps == 1)
			{
				if ((abs(temp_buffer1[x]) > 9) || (abs(temp_buffer2[x]) > 9))
				{
					carry = 1;
				}
			}
			else
			{
				int d = 0;
				for(d = 0; d < reps; ++d)
				{
					if ((x + (512 * d)) < memReq && ((abs(temp_buffer1[x + (512 * d)]) > 9) || (abs(temp_buffer2[x + (512 * d)]) > 9)))
					{
						carry = 1;
					}
				}
			}
		}
		__syncthreads();
	}
	if (x < memReq)
	{
		if (reps == 1)
		{
			if (((x + num1offset) < memReq) && ((x + num2offset) < memReq))
			{
				output[x] = temp_buffer1[x + num1offset] + temp_buffer2[x + num2offset];
			}
			else if ((x + num2offset) < memReq)
			{
				output[x] = temp_buffer2[x + num2offset];
			}
			else if ((x + num1offset) < memReq)
			{
				output[x] = temp_buffer1[x + num1offset];
			}
			else
			{
				//do nothing 
			}
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				if ((((x + (512 * d)) + num1offset) < memReq) && (((x + (512 * d)) + num2offset) < memReq))
				{
					output[(x + (512 * d))] = temp_buffer1[(x + (512 * d)) + num1offset] + temp_buffer2[(x + (512 * d)) + num2offset];
				}
				else if (((x + (512 * d)) + num2offset) < memReq)
				{
					output[(x + (512 * d))] = temp_buffer2[(x + (512 * d)) + num2offset];
				}
				else if (((x + (512 * d)) + num1offset) < memReq)
				{
					output[(x + (512 * d))] = temp_buffer1[(x + (512 * d)) + num1offset];
				}
				else
				{
					//do nothing 
				}
			}
		}
	}		
}

//multiplication<<<1,MAX_THREADS_PER_BLOCK>>>(num1->gpuP, num2->gpuP, num1->length, num2->length, dec1_offset, dec2_offset, temp_buffer1, temp_buffer2, temp_buffer3, output->gpuP, result_req, reps);

__global__ void multiplication(int *num1, int *num2, unsigned int num1Len, unsigned int num2Len, unsigned int num1offset, unsigned int num2offset, int *temp_buffer1, int *temp_buffer2, int *temp_buffer3, int *output2, unsigned int memReq, unsigned int reps)
{
	int x = threadIdx.x;
	if (reps == 1)
	{
		if (x < memReq)
		{
			if (x >= memReq - num1Len)
			{
				temp_buffer1[x] = num1[x - (memReq - num1Len)];
			}
			if (x >= memReq - num2Len)
			{
				temp_buffer2[x] = num2[x - (memReq - num2Len)];
			}
		}
	}
	else
	{
		int d = 0;
		for (d = 0; d < reps; ++d)
		{
			if (d == 0)
			{
				if (x >= memReq - num1Len)
				{
					temp_buffer1[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
					//output2[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
				}
				if (x >= memReq - num2Len)
				{
					temp_buffer2[x + (512 * d)] = num2[x + (512 * d) - (memReq - num2Len)];
				}
			}
			else
			{
				if ((x + (512 * d) - (memReq - num1Len)) < num1Len)
				{
					temp_buffer1[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
					//output2[x + (512 * d)] = num1[x + (512 * d) - (memReq - num1Len)];
					
				}
				if ((x + (512 * d) - (memReq - num2Len)) < num2Len)
				{
					temp_buffer2[x + (512 * d)] = num2[x + (512 * d) - (memReq - num2Len)];
				}
			}
		}
	}
	//move everything to temp buffers
	__shared__ int carry;
	carry = 1;
	__syncthreads();
	while (carry)
	{
		if (reps == 1)
		{
			
			int c1 = 0;
			int c2 = 0;
			if (x < memReq)
			{
				c1 = temp_buffer1[x] / 10;
				c2 = temp_buffer2[x] / 10;
				temp_buffer1[x] %= 10;
				temp_buffer2[x] %= 10;
			}
			__syncthreads();
			if (x < memReq && x != 0)
			{
				temp_buffer1[x - 1] += c1;
				temp_buffer2[x - 1] += c2;
			}
			
		}
		else
		{
			
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				int c1 = 0;
				int c2 = 0;
				if ((x + (512 * d)) < memReq)
				{
					c1 = temp_buffer1[x + (512 * d)] / 10;
					c2 = temp_buffer2[x + (512 * d)] / 10;
					temp_buffer1[x + (512 * d)] %= 10;
					temp_buffer2[x + (512 * d)] %= 10;
				}
				__syncthreads();
				if ((x + (512 * d)) < memReq && (x != 0))
				{
					temp_buffer1[x - 1 + (512 * d)] += c1;
					temp_buffer2[x - 1 + (512 * d)] += c2;
				}
			}
			
		}
		carry = 0;
		__syncthreads();
		if (reps == 1)
		{
			if ((x < memReq) && ((abs(temp_buffer1[x]) > 9) || (abs(temp_buffer2[x]) > 9)))
			{
				carry = 1;
			}
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				if ((x + (512 * d)) < memReq && ((abs(temp_buffer1[x + (512 * d)]) > 9) || (abs(temp_buffer2[x + (512 * d)]) > 9)))
				{
					carry = 1;
				}
			}
		}
		__syncthreads();
	}
	
	//good till here
	//TEST INTITIAL NORMALIZATION 
	
	__shared__ int multCount;
	multCount = 0;
	__syncthreads();
	//output2[x] = reps;
	
	//__syncthreads(); <-- uncomment this too
	//TEST JUST ONE ITERATION
	//while (multCount < num2Len)
	
	while (multCount < num2Len)
	{
		int tempMultCountStore = multCount;
		tempMultCountStore += 1;
		
		if (reps == 1)
		{
			if (x < memReq)
			{
				if (x > multCount)
				{
					temp_buffer3[x - multCount] = temp_buffer2[memReq - multCount - 1] * temp_buffer1[x];
				}
			}
			//check for overflow
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				if (d == 0)
				{
					if ((x > multCount) && ((x + (512 * d)) < memReq))
					{
						temp_buffer3[(x + (512 * d)) - multCount] = temp_buffer2[memReq - multCount - 1] * temp_buffer1[(x + (512 * d))];
					}
				}
				else
				{
					if ((x + (512 * d)) < memReq)
					{
						temp_buffer3[(x + (512 * d)) - multCount] = temp_buffer2[memReq - multCount - 1] * temp_buffer1[(x + (512 * d))];
					}
				}
			}
		}
		
		carry = 0;
		__syncthreads();
		

		int d = 0;
		if (reps == 1)
		{
			for (d = 0; d <= reps; ++d)
			{
				if ((x + (512 * d)) < memReq)
				{
					if (((unsigned int)(temp_buffer3[x + (512 * d)] & ((unsigned int)3 << 30)) > 0) || ((unsigned int)(output2[x + (512 * d)] & ((unsigned int)3 << 30)) > 0))
					{
						carry = 1;
					}
				}
			}
		}
		else
		{
			for (d = 0; d < reps; ++d)
			{
				if ((x + (512 * d)) < memReq)
				{
					if (((unsigned int)(temp_buffer3[x + (512 * d)] & ((unsigned int)3 << 30)) > 0) || ((unsigned int)(output2[x + (512 * d)] & ((unsigned int)3 << 30)) > 0))
					{
						carry = 1;
					}
				}
			}
		}
		__syncthreads();
		
		while (carry)
		{
			if (reps == 1)
			{
				int c1 = 0;
				int c2 = 0;
				if ((reps == 1) && (x < memReq))
				{
					c1 = temp_buffer3[x] / 10;
					c2 = output2[x] / 10;
					temp_buffer3[x] %= 10;
					output2[x] %= 10;
				}
				__syncthreads();
				if (x < memReq && x != 0)
				{
					temp_buffer3[x - 1] += c1;
					output2[x - 1] += c2;
				}
				carry = 0;
				__syncthreads();
				if (x < memReq)
				{
					if ((abs(temp_buffer3[x]) > 9) || abs((output2[x]) > 9))
					{
						carry = 1;
					}
				}
				__syncthreads();
			}
			else
			{
				
				int d = 0;
				for(d = 0; d < reps; ++d)
				{
					int c1 = 0;
					int c2 = 0;
					if ((x + (512 * d)) < memReq)
					{
						c1 = temp_buffer3[x + (512 * d)] / 10;
						c2 = output2[x + (512 * d)] / 10;
						temp_buffer3[x + (512 * d)] %= 10;
						output2[x + (512 * d)] %= 10;
					}
					__syncthreads();
					if (d == 0)
					{
						if ((x != 0) && ((x + (512 * d)) < memReq))
						{
							//SOMEHOW DESYNCRONIZED
							temp_buffer3[x - 1 + (512 * d)] += c1;
							output2[x - 1 + (512 * d)] += c2;
						}
					}
					else
					{
						if (((x + (512 * d)) < memReq))
						{
							//SOMEHOW DESYNCRONIZED
							temp_buffer3[x - 1 + (512 * d)] += c1;
							output2[x - 1 + (512 * d)] += c2;
						}
					}
					__syncthreads();
					carry = 0;
					int d = 0;
					for(d = 0; d < reps; ++d)
					{
						if ((x + (512 * d)) < memReq && ((abs(temp_buffer3[x + (512 * d)]) > 9) || (abs(output2[x + (512 * d)]) > 9)))
						{
							carry = 1;
						}
					}
					__syncthreads();
				}
				
			}
		}
		//perform addition 
		if (reps == 1)
		{
			if (x < memReq)
			{
				output2[x] += temp_buffer3[x];
				temp_buffer3[x] = 0;
			}
			//check for overflow
		}
		else
		{
			int d = 0;
			for(d = 0; d < reps; ++d)
			{
				if ((x + (512 * d)) < memReq)
				{
					output2[x + (512 * d)] += temp_buffer3[x + (512 * d)];
					temp_buffer3[x + (512 * d)] = 0;
				}
			}
		}
		//update counter
		multCount = tempMultCountStore;
		__syncthreads();	
	}
}


//ARITHMETIC FUNCTIONS

bcd* normalize(bcd *num)
{
	unsigned int memReq = NORMALIZATION_EXPANSION + num->length;
	bcd *output = createBcd(memReq);
	zeroBcd(output);
	loadBcdIntoGPU(output);
		
	output->decpos = num->decpos + (memReq - num->length);
	unsigned int reps = (memReq / MAX_THREADS_PER_BLOCK) + 1;
	//printf("reps: %u",reps);
	normalize<<<1,MAX_THREADS_PER_BLOCK>>>(num->gpuP, num->length, output->gpuP,memReq,reps);
	
	return output;
}

bcd* add(bcd *num1, bcd *num2)
{
	//first calc memory requirement for result
	unsigned int result_req = memReqForAddition(num1, num2);
	bcd *output = createBcd(result_req);
	
	if (result_req > temp_buffer_size)
	{
		reallocTempBuffers(result_req * 2);
	}
	
	zeroBcd(output);
	loadBcdIntoGPU(output);
	output->decpos = decimalMovementAddition(num1, num2, result_req);
	unsigned int reps = (result_req / MAX_THREADS_PER_BLOCK) + 1;
	
	//now we'll figure out the decimal offset
	unsigned int decdiff1 = num1->length - num1->decpos;
	unsigned int decdiff2 = num2->length - num2->decpos;
	unsigned int dec1_offset = 0;
	unsigned int dec2_offset = 0;
	
	if (decdiff1 > decdiff2)
	{
		dec2_offset = decdiff1 - decdiff2;
	}
	else
	{
		dec1_offset = decdiff2 - decdiff1;
	}
	zeroTempBuffers();
	addition<<<1,MAX_THREADS_PER_BLOCK>>>(num1->gpuP, num2->gpuP, num1->length, num2->length, dec1_offset, dec2_offset, temp_buffer1, temp_buffer2, output->gpuP, result_req, reps);
	
	return output;
}

bcd *multiply(bcd *num1, bcd *num2)
{
	unsigned int result_req = memReqForMulitiplcation(num1, num2);
	bcd *output = createBcd(result_req);
	
	if (result_req > temp_buffer_size)
	{
		reallocTempBuffers(result_req * 2);
	}
	zeroBcd(output);
	loadBcdIntoGPU(output);
	output->decpos = decimalMovementMultiplication(num1, num2, result_req);
	unsigned int reps = (result_req / MAX_THREADS_PER_BLOCK) + 1;
	//printf("REPS: %u\n", reps);
	
	//now we'll figure out the decimal offset
	unsigned int decdiff1 = num1->length - num1->decpos;
	unsigned int decdiff2 = num2->length - num2->decpos;
	unsigned int dec1_offset = 0;
	unsigned int dec2_offset = 0;
	
	if (decdiff1 > decdiff2)
	{
		dec2_offset = decdiff1 - decdiff2;
	}
	else
	{
		dec1_offset = decdiff2 - decdiff1;
	}
	//printf("RESULT REQ: %u\n", result_req);
	zeroTempBuffers();
	multiplication<<<1,MAX_THREADS_PER_BLOCK>>>(num1->gpuP, num2->gpuP, num1->length, num2->length, dec1_offset, dec2_offset, temp_buffer1, temp_buffer2, temp_buffer3, output->gpuP, result_req, reps);
	
	return output; 
}

//THIS IS A TESTBED FOR OUR LIBRARY 
//AN EXAMPLE
int main()
{
	//bcd *num1 = createBcd(903);
	bcd *num1 = createBcd(7);
	bcd *num2 = createBcd(2);
	
	//bcdFromString("111111111111111111111111111111111111111111111112111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111121111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111.9", num1);
	bcdFromString("112341.9", num1);
	bcdFromString("1.2", num2);
	printBcdNotNormal(num1);
	printBcd(num2);
	
	cudaSetup();
	loadBcdIntoGPU(num1);
	loadBcdIntoGPU(num2);
	bcd* result = multiply(num1, num2); 
	bcd* normResult = normalize(result);
	
	getCompResult(normResult);
	printf("\n");
	printBcd(normResult);
	//printBcd(result);
	
	freeBcd(num1);
	freeBcd(num2);
	freeBcd(result);
	freeBcd(normResult);
	freeTempBuffers();
	return 0;
}