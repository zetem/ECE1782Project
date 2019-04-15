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

/////////////////////////////////////////CPU FUNCTIONS///////////////////////////////////////////////////////////////////
int generate_mv_for_block(PIXEL *currunt_frame, PIXEL *reference_frames, int yb, int xb, int h, int w, int *motion_vector_y, int *motion_vector_x, int *motion_vector_f)    
{    
    int lowest_SAD = 256 * BS * BS;// variable storing the SAD value
    int block_size = BS;
    int y = yb*BS;
    int x = xb*BS;
    // (y,x) the pixel location of the top-left corner of the block.
    
    //Search for the best matching block in the reference frame.
    //The search processes only the block is within the reference frame (not out of boundary).
    for (int ref_index = 0; ref_index < NRF; ref_index++){   
        for (int search_y_radius = MAX(y-SR,0); search_y_radius <= MIN(y+SR,H-BS) ; search_y_radius++){ 
            for (int search_x_radius = MAX(x-SR,0); search_x_radius <= MIN(x+SR,W-BS); search_x_radius++){
                //Calculate SAD of this block with the input block.
                int SAD = 0;
                for(int j =0; j < block_size; j++)
                {
                    for (int i=0; i < block_size; i++)
                    {
                        SAD += abs(currunt_frame[(y+j)*w+x+i] - reference_frames[(NRF-1-ref_index)*w*h+(search_y_radius+j)*w +search_x_radius+i]);
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

void generate_mv_for_frames_cpu(int *motion_vector,PIXEL *luma, int h, int w){
    int nblock_x = w/BS;
    int nblock_y = h/BS;

    PIXEL   *currunt_frame  = (PIXEL *) malloc(h*w*sizeof(PIXEL)); if (currunt_frame == NULL) fprintf(stderr, "Bad malloc on currunt_frame  \n");
    PIXEL   *reference_frames= (PIXEL*) malloc(h*w*NRF*sizeof(PIXEL)); if (reference_frames == NULL) fprintf(stderr, "Bad malloc on reference_frames  \n");

    for (int f =NRF; f < NF; f++)
    {
        memcpy(currunt_frame   ,&luma[h*w*f]      ,h*w    *sizeof(PIXEL));
        memcpy(reference_frames,&luma[h*w*(f-NRF)],h*w*NRF*sizeof(PIXEL));
        for (int y = 0; y < nblock_y; y++){
            for (int x = 0; x < nblock_x; x++){
                int mvy = 0;
                int mvx = 0;
                int mvf = 1;
                generate_mv_for_block(currunt_frame, reference_frames, y, x, h, w, &mvy, &mvx, &mvf);
                motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0] = mvx;
                motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1] = mvy; 
                motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] = mvf;      
            }
        }
    }
    free(currunt_frame );
    free(reference_frames);
}
/////////////////////////////GPU FUNCTIONS///////////////////////////////////////////////////////////////
__global__ void d_generate_mv_one_frame( PIXEL *currunt_frame, PIXEL *reference_frames, int *motion_vector, int h, int w)
{
    __shared__ PIXEL reference_blocks[NRF][BS+2*SR][BS+2*SR]; 
    __shared__ PIXEL currunt_block   [BS][BS]; 
    __shared__ int   sub_result      [NRF][2*SR+1][2*SR+1]; 
    
    //int nblock_y = h/BS;
    int nblock_x = w/BS;
    //cordination of upper left pixel
    int x_block   = blockIdx.x*BS;
    int y_block   = blockIdx.y*BS;
    //these are equal to the motion vector
    int ix = threadIdx.x;     
    int iy = threadIdx.y;
    int iz = threadIdx.z;
    // index to read block of data to search into from reference block
    int idx  = x_block+ix-SR;
    int idy  = y_block+iy-SR;
    // index to read block of data that needs processing
    int idxc = x_block+ix;
    int idyc = y_block+iy;
    // first all threads read the inputs required
    if ((idx >= 0 ) && (idx  < w) && (idy >=0) && (idy < h)) //&& (iy < (BS+2*SR)) && (ix < (BS+2*SR)) )
    {
        reference_blocks[iz][iy][ix] = reference_frames[h*w*(NRF-1-iz)+idy*w+idx];
    }
    // some threads also read current frame
    if (iz == 0 && ix < BS && iy < BS)
    {
        currunt_block[iy][ix] = currunt_frame[idyc*w + idxc];
    }
    __syncthreads();
    // second add all the abs values for each direction of movement(iy, ix) over whole block
    if ((ix < 2*SR+1) && (iy < 2*SR+1))
    {
        sub_result[iz][iy][ix] = 0;
        for (int j = 0; j < BS; j++){
            for (int i =0; i < BS; i++)
            {
                sub_result[iz][iy][ix] += abs(currunt_block[j][i] - reference_blocks[iz][iy+j][ix+i]);        
            }
        }
    }

    __syncthreads();
    // third only one thread compares all SAD and selects the least one and stores motion vector
    if (ix == 0 && iy ==0 && iz == 0) // only one
    {
        int lowest_SAD = 256 * BS * BS;
        // intiallizing motion vector
        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 0 ] = 0;
        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 1 ] = 0;
        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 2 ] = 1;
        for(int z = 0; z< NRF; z++){
            for (int j = 0; j <= 2*SR; j++){
                for (int i =0; i <= 2*SR; i++)
                {
                    if (lowest_SAD > sub_result[z][j][i])
                    {
                        // found lower value so update motion vector
                        lowest_SAD = sub_result[z][j][i];
                        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 0 ] = i  - SR;
                        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 1 ] = j  - SR;
                        motion_vector[(blockIdx.y*nblock_x+blockIdx.x)*3 + 2 ] = z  + 1 ;
                    }
                }
            }
        }
    }
}
// this function invokes gpu kernels
void generate_mv_for_frames_gpu (int *h_motion_vector,PIXEL *luma, int h, int w){

    int nblock_y = h/BS;
    int nblock_x = w/BS;
    PIXEL   *currunt_frame;
    if ( cudaMalloc( (void **) &currunt_frame, h*w*sizeof(PIXEL)) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for currunt_frame\n");
        //exit(EXIT_FAILURE);    
    }
    PIXEL   *reference_frames;
    if ( cudaMalloc( (void **) &reference_frames, h*w*NRF*sizeof(PIXEL)) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for reference_frames\n");
        //exit(EXIT_FAILURE);    
    }
    int *d_motion_vector;
    if ( cudaMalloc( (void **) &d_motion_vector, nblock_y*nblock_x*sizeof(int)*3) != cudaSuccess ){
        fprintf(stderr, "Failed to allocate device vector for d_motion_vector\n");
    }

    for (int f =NRF; f < NF; f++)
    {
        if (cudaMemcpy( currunt_frame, &luma[h*w*f], h*w*sizeof(PIXEL), cudaMemcpyHostToDevice) != cudaSuccess){
            fprintf(stderr, "Failed to copy vector for currunt_frame\n");
        
        }
        if (cudaMemcpy(reference_frames, &luma[h*w*(f-NRF)] ,h*w*NRF*sizeof(PIXEL), cudaMemcpyHostToDevice) != cudaSuccess ) {
            fprintf(stderr, "Failed to copy vector for reference_frames\n");
        }               
        dim3 block(2*SR+BS, 2*SR+BS, NRF);      
        dim3 grid(nblock_x,nblock_y) ;
        d_generate_mv_one_frame<<<grid, block>>>( currunt_frame, reference_frames, d_motion_vector, h, w); 
        cudaDeviceSynchronize() ;

        if (cudaMemcpy(&h_motion_vector[nblock_y*nblock_x*f*3], d_motion_vector,nblock_y*nblock_x*sizeof(int)*3, cudaMemcpyHostToDevice) != cudaSuccess ) {
            fprintf(stderr, "Failed to copy vector for motion_vector \n");
        }      

    }     

    cudaFree(currunt_frame );
    cudaFree(reference_frames);
    cudaFree(d_motion_vector);
}

/////////////////////////SHARED FUNCTIONS///////////////////////////////////
void reconstruct_frames(PIXEL *reconstructed,PIXEL *luma,int *motion_vector, int h, int w){
    int nblock_x = w/BS;
    int nblock_y = h/BS;
    for (int f = NRF; f < NF; f++)
    {
        for (int y = 0; y < nblock_y; y++)
        {
            for (int x = 0; x < nblock_x; x++)
            {
                int mvx = motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0];
                int mvy = motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1];
                int mvf = motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2];
                for(int j =0; j < BS; j++)
                {
                    //copy row by row (size of block)
                    memcpy(&reconstructed[f*W*H + W*(y*BS+j) + x*BS]   ,&luma[(f-mvf)*w*h + w*(mvy+y*BS+j) + mvx+x*BS], BS*sizeof(PIXEL));
                }
            }
        }
    }
}


void read_luma(FILE *fid, PIXEL *luma, PIXEL *crb, int h, int w){
    PIXEL   *temp= (PIXEL*) malloc(H*W*NF*sizeof(PIXEL)); if (temp== NULL) fprintf(stderr, "Bad malloc on temp\n");
    for (int f=0; f<NF; f++) {
        fread (&temp[H*W*f]  ,1,W*H,  fid);
        fread (&crb [W*H/2*f],1,W*H/2,fid);
        //fseek (fid,W*H/2,SEEK_CUR); //seek cb and cr
    }
    for (int f=0; f<NF; f++) {
        for ( int idh =0; idh < H; idh++){
            memcpy(&luma[h*w*f+idh*w] , &temp[H*W*f+idh*W], W*sizeof(PIXEL));
        }
    }
    free(temp);
}

void write_yuv(FILE *fid, PIXEL *reconstructed, PIXEL *crb){
    for (int f=0; f<NF; f++) {
        fwrite(&reconstructed[W*H*f]  ,1,W*H,fid);
        fwrite(&crb          [W*H/2*f],1,W*H/2,fid);
    }
}

void pad_luma(PIXEL *luma,int height,int width){
    for (int i = 0; i < height*width*NF; i++)
        luma[i] = 127;
}

//////////////////////////////////////////////////////////////////////////////

int main()
{
    // take of available refrence frames
    // CIF format
    int nblock_x = W/BS; // (W+(BS-1))/BS;
    int nblock_y = H/BS; // (H+(BS-1))/BS;
    
    int height = nblock_y*BS;
    int width  = nblock_x*BS;

    int number_frames = NF;
    ////////////////////////////////////
    FILE    *fid_in         = fopen("akiyo_cif.yuv","rb");
    FILE    *fid_out      = fopen("akiyo_cif_constructed.yuv","wb");
    FILE    *h_fid_out      = fopen("akiyo_cif_constructed_GPU.yuv","wb");
    int     *h_motion_vector;
    if ( cudaHostAlloc( (void**)&h_motion_vector ,nblock_x*nblock_y*NF*3*sizeof(int), cudaHostAllocWriteCombined) != cudaSuccess ){
        fprintf(stderr, "Bad malloc on h_motion_vector  \n");
    }
    PIXEL   *h_reconstructed= (PIXEL *) malloc(height*width  *NF*sizeof(PIXEL)); if (h_reconstructed == NULL) fprintf(stderr, "Bad malloc on h_reconstructed  \n");
    PIXEL   *luma           = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (luma          == NULL) fprintf(stderr, "Bad malloc on luma           \n");
    int     *motion_vector  = (int *  ) malloc(nblock_x*nblock_y*NF*3    *sizeof(int));   if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    PIXEL   *crb            = (PIXEL *) malloc(H*W/2*number_frames *sizeof(PIXEL)); if (crb           == NULL) fprintf(stderr, "Bad malloc on crb           \n");
    PIXEL   *reconstructed  = (PIXEL *) malloc(H*W*number_frames   *sizeof(PIXEL)); if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");

    pad_luma (luma, height, width);
    read_luma(fid_in,luma,crb,height,width);
    fclose(fid_in);

    double timeStampA = getTimeStamp() ; 

    generate_mv_for_frames_cpu(motion_vector  ,luma, height, width);

    double timeStampB= getTimeStamp() ;

    generate_mv_for_frames_gpu(h_motion_vector,luma, height ,width);

    double timeStampC= getTimeStamp() ;
    
    printf("total CPU time = %.6f\n", timeStampB - timeStampA);
    printf("total GPU time = %.6f\n", timeStampC - timeStampB);
    #ifdef ENABLE_PRINT
    printf("motion_vector\n");
    for (int f =NRF; f < NF; f++)
    {
        for (int y = 0; y < nblock_y; y++)
        {
            for (int x = 0; x < nblock_x; x++)
            {
                if ((motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0]-h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0] != 0) || (motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1]-h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1] != 0) || (motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2]-h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] != 0))
                //if ((motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0] != 0) && (motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1] != 0) && (motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] != 1))
                    printf("-->frame %d at y = %d and x = %d mv = (%d,%d,%d) and h_mv = (%d,%d,%d) \n", f, y*BS, x*BS, 
                              motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2]   , motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1],   motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0], 
                            h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] , h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1], h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0]);
                //else
                //    printf("frame %d at y = %d and x = %d mv = (%d,%d,%d) and h_mv = (%d,%d,%d) \n", f, y*BS, x*BS, 
                //            motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] , motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1], motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0], 
                //            h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+2] , h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+1], h_motion_vector[(nblock_y*nblock_x*f + y*nblock_x + x)*3+0]);
            
            }
        }
    }
    #endif

    reconstruct_frames(reconstructed,luma,motion_vector,height,width);
    write_yuv(fid_out,reconstructed,crb);
    fclose(fid_out);
    ////////DEVICE///////////////
    reconstruct_frames(h_reconstructed,luma,h_motion_vector, height, width);
    write_yuv(h_fid_out,h_reconstructed,crb);
    fclose(h_fid_out);
    //////////////////////////////


    free(luma          );
    free(reconstructed);
    free(h_reconstructed);
    free(motion_vector );
    cudaFreeHost(h_motion_vector);
    free(crb);
}