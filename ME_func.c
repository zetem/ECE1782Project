/*
link to YUV 

*/
#include <cmath>  // for abs
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

//using namespace std;

#define H 1024 // height of each frame
#define W 512 // width of each frame
#define NRF 3 // number of refernce frames

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
} 

void get_Y(FILE *fid, MYTYPE *Yonly, int comps){
    MYTYPE temp;
    for (int ll=0; ll<F; ll++) {
        for (int kk=0; kk<comps; kk++){
            for (int ii=0; ii<M; ii++){
                for (int jj=0; jj<N; jj++){
                    int idx = ll*MN +ii*N +jj;
                    if (kk==0){
                        fread (&Yonly[idx],sizeof(char),1,fid);
                    }
                    else if ((ii%2==0 && jj%2==0) || kk==0){
                        fread (&temp,sizeof(char),1,fid);
                    }
                }
            }
        }
    }
}

//bx by are cordinations for block
// r is search Radius
// i is block size 
int Calculate_SAD(int currunt_frame[H][W], int cur_y, int cur_x, int reference_frame[H][W], int ref_y, int ref_x, int block_size)
{
    int total_diff = 0;
    for(int j =0; j < block_size; j++)
    {
        for (int i=0; i < block_size; i++)
        {
            total_diff += abs(currunt_frame[cur_y+j][cur_x+i] - reference_frame[ref_y+j][ref_x+i])
        }
    }
    return total_diff;
}

int generate_mv_for_block(int currunt_frame[H][W], int reference_frames[NRF][H][W], int y, int x, int r, int i, int& motion_vector_y, int& motion_vector_x)    
{    
    int lowest_SAD = 256 * i * i;// variable storing the SAD value
    
    // (y,x) the pixel location of the top-left corner of the block.
    
    //Search for the best matching block in the reference frame.
    //The search processes only the block is within the reference frame (not out of boundary).
    for (int ref_index = 0; ref_index < NRFl ref_index++)
    {        
        for (int search_x_radius = max(x-r,0); search_x_radius <= min(x+r,W-i); search_x_radius++){
            
            for (int search_y_radius = max(y-r,0); search_y_radius <= min(y+r,H-i) ; search_y_radius++){                
                
                //Calculate SAD of this block with the input block.
                int SAD =  Calculate_SAD(currunt_frame,y,x,referenceFrame[ref_index],search_y_radius,search_x_radius);
                //If this block is better in SAD...
                if (lowestSAD > SAD){
                    lowestSAD = SAD; // Update SAD.
                    motion_vector_x = search_x_radius - x;
                    motion_vector_y = search_y_radius - y; //Update motion vector.
                }
                //If there is a tie in SAD keep last change
            }
        }
    }
    return 1;
}

int main()
{
    // take of available refrence frames
    int block_size = 2;
    int search_radius = 2; 
    motion_vector[H][W][2];
    for (int y = 0; y < H; y+=block_size)
    {
        for (int x = 0; x < W; x+=block_size)
        {
            int mvy = 0;
            int mvx = 0;
            generate_mv_for_block(currunt_frame, reference_frames, y, x, search_radius, block_size, mvy, mvx);
            motion_vector[y][x][0] = mvy;
            motion_vector[y][x][1] = mvx;   
        }
    }
}