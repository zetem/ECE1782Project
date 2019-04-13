/*
link to YUV 
http://trace.eas.asu.edu/yuv/index.html
*/
#include <math.h>  // for abs
#include <stdio.h>
#include <stdint.h> // for uint8_t
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

//using namespace std;
#define PIXEL uint8_t

#define H 288 // height of each frame
#define W 352 // width of each frame
#define NF 200 // number of frames
#define NRF 3 // number of refernce frames
#define BS  4 // block_size
#define SR  2 // search_radius


#define WB W*(BS-1)/

#define ENABLE_PRINT

//because using c not c++
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
} 


int generate_mv_for_block(PIXEL *currunt_frame, PIXEL *reference_frames, int y, int x, int r, int i, int *motion_vector_y, int *motion_vector_x, int *motion_vector_f)    
{    
    int lowest_SAD = 256 * i * i;// variable storing the SAD value
    int block_size = BS;
    // (y,x) the pixel location of the top-left corner of the block.
    
    //Search for the best matching block in the reference frame.
    //The search processes only the block is within the reference frame (not out of boundary).
    for (int ref_index = 0; ref_index < NRF; ref_index++)
    {   
        for (int search_x_radius = MAX(x-r,0); search_x_radius <= MIN(x+r,W-i); search_x_radius++){
            
            for (int search_y_radius = MAX(y-r,0); search_y_radius <= MIN(y+r,H-i) ; search_y_radius++){                
                
                //Calculate SAD of this block with the input block.
                //int SAD =  Calculate_SAD(currunt_frame,y,x,referenceFrame[ref_index],search_y_radius,search_x_radius);
                int SAD = 0;
                for(int j =0; j < block_size; j++)
                {
                    for (int i=0; i < block_size; i++)
                    {
                        SAD += abs(currunt_frame[(y+j)*W+x+i] - reference_frames[(NRF-1-ref_index)*W*H+(search_y_radius+j)*W +search_x_radius+i]);
                    }
                }
                //If this block is better in SAD...
                if (lowest_SAD > SAD){
                    lowest_SAD = SAD; // Update SAD.
                    (*motion_vector_x) = search_x_radius - x;
                    (*motion_vector_y) = search_y_radius - y; //Update motion vector.
                    (*motion_vector_f) = ref_index + 1;
                }
                //If there is a tie in SAD keep last change
            }
        }
    }
    return 1;
}


void h_generate_mv_for_frames(int *motion_vector,PIXEL *luma){
    PIXEL   *currunt_frame  = (PIXEL *) malloc(H*W    *sizeof(PIXEL)); if (currunt_frame == NULL) fprintf(stderr, "Bad malloc on currunt_frame  \n");
    PIXEL   *reference_frames= (PIXEL*) malloc(H*W*NRF*sizeof(PIXEL)); if (reference_frames == NULL) fprintf(stderr, "Bad malloc on reference_frames  \n");
    for (int f =NRF; f < NF; f++)
    {
        memcpy(currunt_frame   ,&luma[H*W*f]      ,H*W    *sizeof(PIXEL));
        memcpy(reference_frames,&luma[H*W*(f-NRF)],H*W*NRF*sizeof(PIXEL));
        for (int y = 0; y < H; y+=BS)
        {
            for (int x = 0; x < W; x+=BS)
            {
                int mvy = 0;
                int mvx = 0;
                int mvf = 1;
                generate_mv_for_block(currunt_frame, reference_frames, y, x, SR, BS, &mvy, &mvx, &mvf);
                motion_vector[(W*H*f + y*W + x)*3+0] = mvy;
                motion_vector[(W*H*f + y*W + x)*3+1] = mvx; 
                motion_vector[(W*H*f + y*W + x)*3+2] = mvf;      
            }
        }
    }    
    free(currunt_frame );
    free(reference_frames);
}

__global__ void d_generate_mv_one_frame( PIXEL *currunt_frame, PIXEL *reference_frames, int *motion_vector)
{
    __shared__ PIXEL reference_blocks[NRF][BS+2*SR][BS+2*SR]; 
    __shared__ PIXEL sub_result[NRF][BS*(2*SR+1)][BS*(2*SR+1)];
    __shared__ PIXEL currunt_block   [BS][BS]; 
    //__device__ int mutex = 0;
    int lowest_SAD = 0; 
    //cordination of upper left pixel
    int x_block = blockIdx.x*BS;
    int y_block = blockIdx.y*BS;
    int block_idx = threadIdx.x/(2*SR+1);
    int block_idy = threadIdx.y/(2*SR+1);
    int ix = threadIdx.x%(2*SR+1);     
    int iy = threadIdx.y%(2*SR+1);
    int iz = threadIdx.z;

    int idx = x_block+threadIdx.x-SR;
    int idy = y_block+threadIdx.y-SR;
    if ((idx >= 0) && (idx < W) && (idy >=0) && (idy < H) && (threadIdx.y < (BS+2*SR)) && (threadIdx.x < (BS+2*SR)) )
    {
        reference_blocks[iz][threadIdx.y][threadIdx.x] = reference_frames[H*W*(NRF-1-iz)+idy*W+idx];
    }
    if ((threadIdx.y < BS) && (threadIdx.x < BS))
    {
        currunt_block[threadIdx.y][threadIdx.x] = currunt_frame[(y_block+threadIdx.y)*W + x_block+threadIdx.x];
    }
    __syncthreads();


    sub_result[iz][threadIdx.y][threadIdx.y] = abs(currunt_block[block_idy][block_idx] - reference_blocks[iz][threadIdx.y][threadIdx.x]);

    __syncthreads();

    if (block_idx == 0 && block_idy ==0)
    {
        for (int i = 0; i < BS; i++){
            for (int j =0; j < BS; j++)
            {
                if ((i != 0) && (j != 0))
                    sub_result[iz][iy][ix] += sub_result[iz][iy+j*(2*SR+1)][ix+i*(2*SR+1)];
            }
        }
    
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y ==0 && threadIdx.z == 0) // only one
    {
        motion_vector[(y_block*W+x_block)*3 + 0 ] = 0;
        motion_vector[(y_block*W+x_block)*3 + 1 ] = 0;
        motion_vector[(y_block*W+x_block)*3 + 2 ] = 1;
        for(int z = 0; z< NRF; z++){
            for (int i = 0; i <= 2*SR; i++){
                for (int j =0; j <= 2*SR; j++)
                {
                    if (lowest_SAD < sub_result[z][j][i])
                    {
                        lowest_SAD = sub_result[z][j][i];
                        motion_vector[(y_block*W+x_block)*3 + 0 ] = i - SR;
                        motion_vector[(y_block*W+x_block)*3 + 1 ] = j - SR;
                        motion_vector[(y_block*W+x_block)*3 + 2 ] = iz+1;
                    }
                }
            }
        }
    }
    

    /*    while (0 != (atomicCAS(&mutex, 0, 1)));        
        if (sub_result[iz][iy][ix] < lowest_SAD)
        {
            lowest_SAD = sub_result[iz][iy][ix];
            motion_vector[(y_block*W+x_block)*3 + 0 ] = ix;
            motion_vector[(y_block*W+x_block)*3 + 1 ] = iy;
            motion_vector[(y_block*W+x_block)*3 + 2 ] = iz;

        }
        atomicExch(mutex, 0);//unlock
    } 
    */ 
    __syncthreads();
}

void d_generate_mv_for_frames (int *h_motion_vector,PIXEL *luma){

    PIXEL   *currunt_frame;
    if ( cudaMalloc( (void **) &currunt_frame, H*W*sizeof(PIXEL)) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for currunt_frame\n");
        //exit(EXIT_FAILURE);    
    }
    PIXEL   *reference_frames;
    if ( cudaMalloc( (void **) &reference_frames, H*W*NRF*sizeof(PIXEL)) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for reference_frames\n");
        //exit(EXIT_FAILURE);    
    }
    int *d_motion_vector;
    if ( cudaMalloc( (void **) &d_motion_vector, H*W*sizeof(int)*3) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for d_motion_vector\n");
    }

    for (int f =NRF; f < NF; f++)
    {
        if (cudaMemcpy( currunt_frame, &luma[H*W*f], H*W*sizeof(PIXEL), cudaMemcpyHostToDevice) != cudaSuccess){
            fprintf(stderr, "Failed to copy vector for currunt_frame\n");
        
        }
        if (cudaMemcpy(reference_frames, &luma[H*W*(f-NRF)] ,H*W*NRF*sizeof(PIXEL), cudaMemcpyHostToDevice) != cudaSuccess ) {
            fprintf(stderr, "Failed to copy vector for reference_frames\n");
        }               
        dim3 block(NRF, BS*(2*SR+1), BS*(2*SR+1));      
        dim3 grid( (W + BS-1)/BS, (H + BS -1)/BS) ;
        d_generate_mv_one_frame<<<grid, block>>>( currunt_frame, reference_frames, d_motion_vector); 
        cudaDeviceSynchronize() ;

        if (cudaMemcpy(&h_motion_vector[H*W*f], d_motion_vector,H*W*sizeof(int)*3, cudaMemcpyHostToDevice) != cudaSuccess ) {
            fprintf(stderr, "Failed to copy vector for motion_vector \n");
        }      

    }     

    cudaFree(currunt_frame );
    cudaFree(reference_frames);
    cudaFree(d_motion_vector);
}

void reconstruct_frames(PIXEL *reconstructed,PIXEL *luma,int *motion_vector){
    for (int f = NRF; f < NF; f++)
    {
        for (int y = 0; y < H; y+=BS)
        {
            for (int x = 0; x < W; x+=BS)
            {
                int mvy = motion_vector[(W*H*f + y*W + x)*3+0];
                int mvx = motion_vector[(W*H*f + y*W + x)*3+1];
                int mvf = motion_vector[(W*H*f + y*W + x)*3+2];
                for(int j =0; j < BS; j++)
                {
                    //copy row by row (size of block)
                    memcpy(&reconstructed[f*W*H + W*(y+j) + x]   ,&luma[(f-mvf)*W*H + W*(mvy+y+j) + mvx+x], BS*sizeof(PIXEL));
                }
            }
        }
    }
}

void read_luma(FILE *fid, PIXEL *luma, PIXEL *crb){
    for (int f=0; f<NF; f++) {
        fread (&luma[W*H *f],1,W*H,  fid);
        fread (&crb[W*H/2*f],1,W*H/2,fid);
        //fseek (fid,W*H/2,SEEK_CUR); //seek cb and cr
    }
}

void write_yuv(FILE *fid, PIXEL *reconstructed, PIXEL *crb){
    for (int f=0; f<NF; f++) {
        fwrite(&reconstructed[W*H*f]  ,1,W*H,fid);
        fwrite(&crb          [W*H/2*f],1,W*H/2,fid);
    }
}

int main()
{
    // take of available refrence frames
    int block_size = BS; //min can be 2
    //int search_radius = SR; 
    // CIF format
    int height = H; int width = W;
    int number_frames = NF;
    //int number_block_x = width/block_size;
    //int number_block_y = height/block_size;
    ////////////////////////////////////
    FILE    *fid_in         = fopen("akiyo_cif.yuv","rb");
    FILE    *h_fid_out      = fopen("akiyo_cif_constructed.yuv","wb");
    FILE    *d_fid_out      = fopen("akiyo_cif_constructed_GPU.yuv","wb");
    PIXEL   *luma           = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (luma          == NULL) fprintf(stderr, "Bad malloc on luma           \n");
    int     *motion_vector  = (int *  ) malloc(height*width*number_frames*sizeof(int)*3); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    int     *h_motion_vector;
    if ( cudaHostAlloc( (void**)&h_motion_vector ,height*width*number_frames*sizeof(int)*3, cudaHostAllocWriteCombined) != cudaSuccess ){
        fprintf(stderr, "Bad malloc on h_motion_vector  \n");
    }
// variables for test
    PIXEL   *crb            = (PIXEL *) malloc(height*width/2*number_frames*sizeof(PIXEL)); if (crb           == NULL) fprintf(stderr, "Bad malloc on crb           \n");
    PIXEL   *h_reconstructed= (PIXEL *) malloc(height*width*number_frames  *sizeof(PIXEL)); if (h_reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    PIXEL   *d_reconstructed= (PIXEL *) malloc(height*width*number_frames  *sizeof(PIXEL)); if (d_reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");

    read_luma(fid_in,luma,crb);
    fclose(fid_in);

    double timeStampA = getTimeStamp() ; 

    h_generate_mv_for_frames(motion_vector,luma);

    double timeStampB= getTimeStamp() ;

    d_generate_mv_for_frames(h_motion_vector,luma);

    double timeStampC= getTimeStamp() ;
    

    printf("total CPU time = %.6f\n", timeStampB - timeStampA);
    printf("total GPU time = %.6f\n", timeStampC - timeStampB);
     
    #ifdef ENABLE_PRINT
    /// test results
    printf("motion_vector\n");
    for (int f =NRF; f < number_frames; f++)
    {
        for (int y = 0; y < H; y+=block_size)
        {
            for (int x = 0; x < W; x+=block_size)
            {
                if ((motion_vector[(W*H*f + y*W + x)*3+0]-h_motion_vector[(W*H*f + y*W + x)*3+0] != 0) || (motion_vector[(W*H*f + y*W + x)*3+1]-h_motion_vector[(W*H*f + y*W + x)*3+1] != 0) || (motion_vector[(W*H*f + y*W + x)*3+2]-motion_vector[(W*H*f + y*W + x)*3+2] != 0))
                    printf("frame %d at y = %d and x = %d mv = (%d,%d,%d) and h_mv = (%d,%d,%d) \n", f, y, x, motion_vector[(W*H*f + y*W + x)*3+0] , motion_vector[(W*H*f + y*W + x)*3+1], motion_vector[(W*H*f + y*W + x)*3+2], h_motion_vector[(W*H*f + y*W + x)*3+0] , h_motion_vector[(W*H*f + y*W + x)*3+1], h_motion_vector[(W*H*f + y*W + x)*3+2]);
            }
        }
    }
    #endif

    reconstruct_frames(h_reconstructed,luma,motion_vector);
    write_yuv(h_fid_out,h_reconstructed,crb);
    fclose(h_fid_out);
    ////////DEVICE///////////////
    reconstruct_frames(d_reconstructed,luma,h_motion_vector);
    write_yuv(d_fid_out,d_reconstructed,crb);
    fclose(d_fid_out);
    //////////////////////////////


    free(luma          );
    free(d_reconstructed);
    free(h_reconstructed);
    free(motion_vector );
    cudaFreeHost(h_motion_vector);
    free(crb);
}