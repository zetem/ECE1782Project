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
#include <omp.h>



//using namespace std;
#define PIXEL uint8_t

#define H 288 // height of each frame
#define W 352 // width of each frame
#define NF 200 // number of frames
#define NRF 3 // number of refernce frames
#define BS  4 // block_size
#define SR  2 // search_radius

//#define ENABLE_PRINT

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
    
    int ref_index, search_x_radius, search_y_radius;

    #pragma omp parallel for private(ref_index)
    for (ref_index = 0; ref_index < NRF; ref_index++)
    {
        for (search_x_radius = MAX(x-r,0); search_x_radius <= MIN(x+r,W-i); search_x_radius++){

            for (search_y_radius = MAX(y-r,0); search_y_radius <= MIN(y+r,H-i) ; search_y_radius++){
                
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
    int search_radius = SR; 
    // CIF format
    int height = H; int width = W;
    int number_frames = NF;
    int number_block_x = width/block_size;
    int number_block_y = height/block_size;
    ////////////////////////////////////
    FILE    *fid_in         = fopen("akiyo_cif.yuv","rb");
    FILE    *fid_out        = fopen("akiyo_cif_constructed.yuv","wb");
    PIXEL   *luma           = (PIXEL *) malloc(height*width*number_frames*sizeof(PIXEL)); if (luma          == NULL) fprintf(stderr, "Bad malloc on luma           \n");
    int     *motion_vector  = (int *  ) malloc(height*width*number_frames*sizeof(int)*3); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    PIXEL   *currunt_frame  = (PIXEL *) malloc(height*width              *sizeof(PIXEL)); if (currunt_frame == NULL) fprintf(stderr, "Bad malloc on currunt_frame  \n");
    PIXEL   *reference_frames= (PIXEL*) malloc(height*width*NRF          *sizeof(PIXEL)); if (reference_frames == NULL) fprintf(stderr, "Bad malloc on reference_frames  \n");
// variables for test
    PIXEL   *crb            = (PIXEL *) malloc(height*width/2*number_frames*sizeof(PIXEL)); if (crb           == NULL) fprintf(stderr, "Bad malloc on crb           \n");
    PIXEL   *reconstructed  = (PIXEL *) malloc(height*width*number_frames  *sizeof(PIXEL)); if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");

    read_luma(fid_in,luma,crb);
    fclose(fid_in);

    double timeStampA = getTimeStamp() ; 

    int f, x, y;
    int mvy, mvx, mvf;
    #pragma omp parallel for private(f)
    for (f=NRF; f < number_frames; f++)
    {
        memcpy(currunt_frame   ,&luma[height*width*f]      ,height*width    *sizeof(PIXEL));
        memcpy(reference_frames,&luma[height*width*(f-NRF)],height*width*NRF*sizeof(PIXEL));
        #pragma omp parallel for private(y)
        for (y = 0; y < H; y+=block_size)
        {
            #pragma omp parallel for private(x)
            for (x = 0; x < W; x+=block_size)
            {
                mvy = 0;
                mvx = 0;
                mvf = 1;
                generate_mv_for_block(currunt_frame, reference_frames, y, x, search_radius, block_size, &mvy, &mvx, &mvf);
                motion_vector[(W*H*f + y*W + x)*3+0] = mvy;
                motion_vector[(W*H*f + y*W + x)*3+1] = mvx; 
                motion_vector[(W*H*f + y*W + x)*3+2] = mvf;      
            }
        }
    }

    double timeStampB= getTimeStamp() ;
    printf("total CPU time = %.6f\n", timeStampB - timeStampA); 
    /// test results
    printf("motion_vector\n");

    #ifdef ENABLE_PRINT
    for (int f =NRF; f < number_frames; f++)
    {
        for (int y = 0; y < H; y+=block_size)
        {
            for (int x = 0; x < W; x+=block_size)
            {
                printf("frame %d at y = %d and x = %d mv = (%d,%d,%d)\n", f, y, x, motion_vector[(W*H*f + y*W + x)*3+0] , motion_vector[(W*H*f + y*W + x)*3+1], motion_vector[(W*H*f + y*W + x)*3+2]);
            }
        }
    }
    #endif

    reconstruct_frames(reconstructed,luma,motion_vector);
    write_yuv(fid_out,reconstructed,crb);
    fclose(fid_out);

    free(luma          );
    free(reconstructed );
    free(motion_vector );
    free(currunt_frame );
    free(reference_frames);
    free(crb);
}
