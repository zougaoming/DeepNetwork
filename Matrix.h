/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef _MATRIX_H
#define _MATRIX_H
/* Exported types ------------------------------------------------------------*/

typedef struct Dshape{
    int shape[4]; //最多四维
}Dshape;

typedef struct Matrix{
    double *array;
    Dshape dshape; //数组结构
    int length; //长度
    int size; //空间大小
}Matrix;

typedef struct MatrixResult{
    double *array;
    int length; //长度
}MatrixResult;

Matrix *creatAsMatrixFromDatas(double *data,int data_len, Dshape dshape);
Matrix *creatMatrixFromDatas(double *data,int data_len, Dshape dshape);
Matrix *creatMatrixFromValue(double value, Dshape dshape);
Matrix *creatMatrixFromArange(double startVal, double stepVal,Dshape dshape);
Matrix *creatMatrixFromLinspace(double startVal, double endVal,Dshape dshape);
Matrix *creatZerosMatrix(Dshape dshape);
Matrix *creatOnesMatrix(Dshape dshape);
Matrix *creatIdentitySecondOrderMatrix(Dshape dshape);
Matrix * zeros_like(Matrix *m);
Matrix * ones_like(Matrix *m);
void initDshape(Dshape *dshape,int *shapeval);
void initDshapeInt(Dshape *dshape,int s_1,int s_2,int s_3,int s_4);
int reshape(Matrix *m,Dshape dshape);
void clearMatrix(Matrix *m);
void printShape(Matrix *m);

int getMatrixNdim(Matrix *m);
void printarray(Matrix *m);
//获取二维数组，最后两维
void get2dim(Matrix *m,Matrix *mto,int dimen0,int dimen1);
int PandingMatrix4D(Matrix *m,unsigned zeropanding);
int setZeros(Matrix *m);
void getSecondOrderSubMatrix2(Matrix *m,Matrix* mto,int startRow,int startColume);
double getSecondOrderMatrixTrace(Matrix *m);
int getMatrixElem(Matrix *m,int dimen0,int dimen1,int dimen2,int dimen3,double *elem);
int modifyMatrixElem(Matrix *m,int dimen0,int dimen1,int dimen2,int dimen3,double elem);
Matrix *copyMatrix(Matrix *m);
//比较两个数组的维度是否一致
int compareMatrix_Shape(Matrix* m1,Matrix *m2);
int compareMatrix(Matrix *m1, Matrix *m2);
Matrix *getSecondOrderMatrixRows(Matrix *m,int startRow,int endRow);
Matrix *getSecondOrderMatrixColumes(Matrix *m,int startColume,int endColume);
Matrix *getSecondOrderSubMatrix(Matrix *m,int startRow,int startColume,int endRow,int endColume);
Matrix *getSecondOrderLeftSubMatrix(Matrix *m,int row,int colume);
int transposeSecondOrderMatrix(Matrix *m);
int isSymmetricMatrix(Matrix *m);
int swapSecondOrderMatrixRow(Matrix *m, int row1,int row2);
int swapSecondOrderMatrixColume(Matrix *m, int colume1,int colume2);
int kAddSecondOrderMatrixRow(Matrix *m, int row,double k);
int kSubSecondOrderMatrixRow(Matrix *m, int row,double k);
int kMulSecondOrderMatrixRow(Matrix *m, int row,double k);
int kDivSecondOrderMatrixRow(Matrix *m, int row,double k);
int addSecondOrderMatrixRows(Matrix *m, int row1,int row2);
int subSecondOrderMatrixRows(Matrix *m, int row1,int row2);
int mulSecondOrderMatrixRows(Matrix *m, int row1,int row2);
int divSecondOrderMatrixRows(Matrix *m, int row1,int row2);
int kAddSecondOrderMatrixColume(Matrix *m, int colume,double k);
int kSubSecondOrderMatrixColume(Matrix *m, int colume,double k);
int kMulSecondOrderMatrixColume(Matrix *m, int colume,double k);
int kDivSecondOrderMatrixColume(Matrix *m, int colume,double k);
int addSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2);
int subSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2);
int mulSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2);
int divSecondOrderMatrixColumes(Matrix *m, int colume1, int colume2);
int deleteSecondOrderMatrixRows(Matrix *m,int startRow,int endRow);
int deleteSecondOrderMatrixColumes(Matrix *m,int startColume,int endColume);
int deleteSecondOrderMatrixRowAndColume(Matrix *m,int row,int colume);
int spliceSecondOrderMatrixRow(Matrix *m1,Matrix *m2);
int spliceSecondOrderMatrixColume(Matrix *m1,Matrix *m2);
int kAddMatrix(Matrix *m,double k);
int kSubMatrix(Matrix *m,double k);
int kMulMatrix(Matrix *m,double k);
int kDivMatrix(Matrix *m,double k);
Matrix *addSecondOrderMatrixs(Matrix *m1,Matrix *m2);
void addSecondOrderMatrixs2(Matrix *m1,Matrix *m2);//不重新申请内存，直接保存到M1中。
void addSecondOrderMatrixsby2d(Matrix *m1,Matrix *m2,int dim0,int dim1);//增加到最后两维
Matrix *subSecondOrderMatrixs(Matrix *m1,Matrix *m2);
Matrix *dotSecondOrderMatrixs(Matrix *m1,Matrix *m2);
void dotSecondOrderMatrixs2(Matrix *m1,Matrix *m2);//两个数组点积,对应元素相乘,m1,m2的维数必须相同(支持4维度),不分配内存，保存到m1
Matrix *mulSecondOrderMatrixs(Matrix *m1,Matrix *m2);
int detSquareMatrixs(Matrix *m,double *result);
int getSquareMatrixElemAlgebraicComplement(Matrix *m,int row,int colume,double *result);
Matrix *getSquareMatrixRawAlgebraicComplement(Matrix *m,int row);
Matrix *getSquareMatrixAdjointMatrix(Matrix *m);
Matrix *invSquareMatrix(Matrix *m);
Matrix *getEchelonMatrix(Matrix *m);
int getSecondOrderMatrixRank(Matrix *m ,int *rank);
Matrix *solveHomoLinearEquations(Matrix *A);
Matrix *solveNonHomoLinearEquations(Matrix *A, Matrix *B);
Matrix *getSymmetricMatrixEigen(Matrix *m);
double getMatrixInfNorm(Matrix *m);
double getMatrixL0Norm(Matrix *m);
double getMatrixL1Norm(Matrix *m);
double getMatrixL2Norm(Matrix *m);
double getMatrixL21Norm(Matrix *m);
double getMatrixSum(Matrix *m);//获取每个元素之和
int setMatrixArray(Matrix *m1,Matrix *m2,int dimen0,int dimen1);//第二个数组合并到第一个数组的最后二维
void destroyMatrix(Matrix *m);
void getMaximumMatrix(Matrix *m,double k);//数组中如果大于K，则为本身，如果小于K，则为K
int rot90Matrix(Matrix *m,int d);//顺时针旋转矩阵
//求往后算ndim维度的和
double getMatrixSumbyDim(Matrix *m,int ndim,int dim0,int dim1);
#endif