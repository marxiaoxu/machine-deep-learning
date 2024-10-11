#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX_LENTH_OF_LINE 1024 //文本行最大容量1024

/**************修改如下值*******************/
#define In 10    //输入层In个节点
#define Hide 12  //隐藏层Hide个节点
#define Out 1   //输出层Out个节点
#define N 346    //每次训练N组数据96   292 
#define Ntest 8    //每次测试Ntest组数据
#define Err_max  0.000001 //允许的误差精度
#define rate  0.001 //学习速率
#define In_divisor  100.0 //输入除数 保持所有输入数据在-1~1内
#define Out_divisor  1.0 //输出除数 保持所有输出数据在-1~1内
int Times = 100000; //训练次数
/******************************************/

double weight_IH[In][Hide];  //输入层隐藏层间权值矩阵 IH = Input -> Hide
double weight_HO[Hide][Out]; //隐藏层输出层间权值矩阵 OH = Output -> Hide
double bias_H[Hide]; //隐藏层偏置
double bias_O[Out];  //输出层偏置
double delta_IH[N][Hide]; //输入层到隐藏层 修正值
double delta_HO[N][Out];  //隐藏层到输出层 修正值

double error[N];    //一组样本误差
double Err_Sum = 0; //一次训练总误差 及 每组样本误差和
 
double train_In[N][In];   //训练输入值
double train_Out[N][Out]; //训练输出值
double hide_Out[N][Hide]; //隐藏层输出
double BP_Out[N][Out];  //BP网络最终输出

FILE* fpstr; //定义文件指针

//设置行光标
static void SetPositionByLine(FILE *fp, int nLine)
{
	int i = 0;
	char buffer[MAX_LENTH_OF_LINE];
    fseek(fp, 0, SEEK_SET);     //定位到开头
	for (i = 0; i < nLine; i++) {
		fgets(buffer, MAX_LENTH_OF_LINE, fp);
    }
}

//从文本获取数据 文本名 首地址 行数 列数
static void getTxtData(char *FileName, double *Data, int Row, int Column)
{
    int i, j;
    FILE *fp;
	fp = fopen(FileName, "r");
    if(fp == NULL) {
        printf("读取失败\r\n");
        return;
    }    
    for(i = 0; i < Row; i++) {
        SetPositionByLine(fp, i); //文本光标转至第i行
        for(j = 0; j < Column; j++) {
        fscanf(fp, "%lf ", Data + i * Column + j); //读数据
        }
    }
    fclose(fp);
}

//保存数据至文本 文本名 首地址 行数 列数
static void saveTxtData(char *FileName, double *Data, int Row, int Column)
{
    int i, j;
    FILE *fp;
	fp = fopen(FileName, "w+");
    if(fp == NULL) {
        printf("创建失败\r\n");
        return;
    }    
    for(i = 0; i < Row; i++) {
        for(j = 0; j < Column; j++) {
        fprintf(fp, "%lf ", *(Data + i * Column + j)); //写数据
        }
        fprintf(fp, "\n"); //换行     
    }
    fclose(fp);
}

//参数初始化
static void parm_init() 
{
    int i,j;
    srand((unsigned)time(NULL));    //撒种子生成随机数
    //初始化 输入层到隐藏层 权值
    for(i = 0; i < In; i++) {
        for(j = 0; j < Hide; j++) {
            weight_IH[i][j] = (double)(rand() % 20 - 10) / 10; //取随机数-1.0 ~ +1.0
        }
    }
    //初始化 隐藏层 偏置  
    for(i = 0; i < Hide; i++) {
            bias_H[i] = (double)(rand() % 10) / 10; //取随机数0.0 ~ +1.0
    }
    //初始化 隐藏层到输出层 权值
    for(i = 0; i < Hide; i++) {
        for(j = 0; j < Out; j++) {
            weight_HO[i][j] = (double)(rand() % 20 - 10) / 10; //取随机数-1.0 ~ +1.0
        }
    }
    //初始化 输出层 偏置 
    for(i = 0; i < Out; i++) {
            bias_O[i] = (double)(rand() % 10) / 10; //取随机数0.0 ~ +1.0
    }
    for(i = 0; i < N; i++) {
        error[i] = 0; //初始化一组样本总误差 
        for(j = 0; j < Hide; j++) {        
            delta_IH[i][j] = 0; //输入层到隐藏层 修正值
        } 
        for(j = 0; j < Out; j++) {
            delta_HO[i][j] = 0; //隐藏层到输出 修正值
        }         
    }
    
}

//激活函数
static double sigmoid(double x)
{
    double tmp = 1.0 / (1.0 + exp(-x));
    return tmp;
}
 
//获取单节点输出 数据首地址 数据列数 权值首地址 权值列数 偏置
static double getOneNodeValue(double *value, int valueColumn, double *weight, int weightColumn, double bias)
{
    int i = 0 ;double tmp = 0.0;
    for (i = 0 ; i < valueColumn ; i++)
        tmp += *(value+i) * *(weight+i * weightColumn); //x1*w1 + x2*w2 + x3*w3 + ....
    tmp -= bias; //偏置
    return sigmoid(tmp);
}

//获取训练数据 输入输出
static void getTrainData()
{
    int i,j;
    //读取文本，获取样本
    getTxtData("trainDataIn.txt", *train_In, N, In);   //输入值
    getTxtData("trainDataOut.txt", *train_Out, N, Out);  //输出值
    for(i = 0; i < N; i++) {
        for(j = 0; j < In; j++) {
            //printf("%lf ", train_In[i][j]);     //打印样本
            train_In[i][j] /= In_divisor;       //保持所有输入数据在-1~1内       
        }    
        for(j = 0; j < Out; j++) {
            //printf("%lf ", train_Out[i][j]);     //打印样本
            train_Out[i][j] /= Out_divisor;      //保持所有输出数据在-1~1内         
        }            
        //printf("\r\n");
     }
}

//训练times次
static void train(int times)
{
    int i,j,k,n;
    for(n = 0; n < times; n++)
    {
        Err_Sum = 0;
        double delta_err[N] = {0};
        //正向传输
        for(i  = 0; i < N; i++) {
            //计算隐藏层输出 in -> hide
            for(j = 0; j < Hide; j++) {
                hide_Out[i][j] = getOneNodeValue(train_In[i], In, *weight_IH+j, Hide, bias_H[j]);
            }
            //计算输出层输出 hide -> out
            for(j = 0; j < Out; j++) {
                BP_Out[i][j] = getOneNodeValue(hide_Out[i], Hide, *weight_HO+j, Out, bias_O[j]);
                if((times - n) == 1) 
                printf("%lf \r\n",BP_Out[i][j]*Out_divisor); //最后一次训练打印每组输出
            }
            //计算样本误差 
            for(j = 0; j < Out; j++) {
                delta_err[i] = train_Out[i][j] - BP_Out[i][j]; //误差
                error[i] += delta_err[i] * delta_err[i];  //误差平方和
            }
            error[i] /= 2;    
            //计算隐藏层输出层间权值调整系数  有公式
            for(j = 0; j < Out; j++) {
                delta_HO[i][j] = delta_err[i] * BP_Out[i][j] * (1-BP_Out[i][j]);
            }
            //计算输入层到隐藏层的权值调整系数  有公式
            for(j = 0; j < Hide; j++) {               
                delta_IH[i][j] = 0; //清0
                for(k = 0; k < Out; k++) {
                    delta_IH[i][j] += delta_HO[i][k] * weight_HO[j][k] * hide_Out[i][j] * (1- hide_Out[i][j]);                       
                }  
            }
            Err_Sum += error[i]; //计算
        }       
        //反向传输
        //调整weight_IH
        double temp = 0.0;
        for(i = 0; i < In; i++) {
            for(j = 0;j < Hide; j++) {
                temp = 0;
                for(k = 0; k < N; k++) {
                    temp += delta_IH[k][j] * train_In[k][i];
                }
                weight_IH[i][j] += rate*temp;
                                
            }
        }
        //调整bias_H
        for(i = 0; i < Hide; i++) {
            temp = 0;
            for(j = 0; j < N; j++) {
                temp -= delta_IH[j][i];
            }
            bias_H[i] += rate * temp;
        }
        //调整weight_HO
        for(i = 0; i < Hide; i++) {
            for(j = 0;j < Out; j++) {
                temp = 0;
                for (k = 0; k < N; k++) {
                    temp += delta_HO[k][j] * hide_Out[k][i];
                }
                weight_HO[i][j] += rate * temp;
            }
        }
        //调整bias_O
        for (i = 0; i < Out; i++) {
            temp = 0;
            for(j = 0;j < N; j++) {
                temp -= delta_HO[j][i];
            }
            bias_O[i] += rate * temp;
        }
        if(Err_Sum < Err_max) break; //误差到达允许值 跳出循环 停止训练
        if((n % (times / 10)) == 0) { //每Times/10次打印一次信息
        printf("次数:%d 误差:%lf \r\n", n + times / 10, Err_Sum); 
        //printf("----------------------------------------------------------\r\n");
        }
    }
}

//输入(处理后的 / In_divisor)测试数据首地址 输出数据首地址
static void test(double *data, double *output)
{
    int i;
    double hide[Hide] = {0};
    for(i=0;i<Hide;i++) {
        hide[i] = getOneNodeValue(data, In, *weight_IH+i, Hide, bias_H[i]); 
    }
    for(i=0;i<Out;i++) {
        output[i] = getOneNodeValue(hide, Hide, *weight_HO+i, Out, bias_O[i]) * Out_divisor; 
    }
}
 
int main()
{
    int i, j, mod = 0;
    double testData[Ntest][In];
    double OutData[Ntest][Out];
    parm_init(); //参数初始化
    printf("是否读取参数 1: 读取 0: 不读取\r\n");
    scanf("%d", &mod);
    if(mod == 1) {
        //读取参数
        getTxtData("weight_IH.txt", *weight_IH, In, Hide);
        getTxtData("weight_HO.txt", *weight_HO, Hide, Out);
        getTxtData("bias_H.txt", bias_H, Hide, 1);
        getTxtData("bias_O.txt", bias_O, Out, 1);
    }

    printf("是否训练 1: 训练 0: 不训练\r\n");
    scanf("%d", &mod);
    if(mod == 1) {
        getTrainData();    
        printf("训练次数：\r\n");
        scanf("%d", &Times);    
        train(Times);        
        printf("-------------------------训练完成-------------------------\r\n" );        
        printf("\r\n训练完毕 误差:%lf\r\n", Err_Sum);
    }

    getTxtData("testData.txt", *testData, Ntest, In); //获取测试数据
    for(i = 0; i < In; i++) {
        for(j = 0; j < Ntest; j++) {
         testData[j][i] /= In_divisor;  //处理输入
        } 
    }

    printf("\r\n--------------------------测试集--------------------------\r\n" );
    for(i = 0; i < Ntest; i++) { 
        test(testData[i],OutData[i]);
        for(j = 0; j < Out; j++) {
            printf("%lf ", OutData[i][j]);                
        }
        printf("\r\n");   
    }
    //存储参数
    saveTxtData("weight_IH.txt", *weight_IH, In, Hide);
    saveTxtData("weight_HO.txt", *weight_HO, Hide, Out);
    saveTxtData("bias_H.txt", bias_H, Hide, 1);
    saveTxtData("bias_O.txt", bias_O, Out, 1);
          
    return 0;
}
