#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX_LENTH_OF_LINE 1024 //�ı����������1024

/**************�޸�����ֵ*******************/
#define In 10    //�����In���ڵ�
#define Hide 12  //���ز�Hide���ڵ�
#define Out 1   //�����Out���ڵ�
#define N 346    //ÿ��ѵ��N������96   292 
#define Ntest 8    //ÿ�β���Ntest������
#define Err_max  0.000001 //���������
#define rate  0.001 //ѧϰ����
#define In_divisor  100.0 //������� ������������������-1~1��
#define Out_divisor  1.0 //������� �����������������-1~1��
int Times = 100000; //ѵ������
/******************************************/

double weight_IH[In][Hide];  //��������ز��Ȩֵ���� IH = Input -> Hide
double weight_HO[Hide][Out]; //���ز�������Ȩֵ���� OH = Output -> Hide
double bias_H[Hide]; //���ز�ƫ��
double bias_O[Out];  //�����ƫ��
double delta_IH[N][Hide]; //����㵽���ز� ����ֵ
double delta_HO[N][Out];  //���ز㵽����� ����ֵ

double error[N];    //һ���������
double Err_Sum = 0; //һ��ѵ������� �� ÿ����������
 
double train_In[N][In];   //ѵ������ֵ
double train_Out[N][Out]; //ѵ�����ֵ
double hide_Out[N][Hide]; //���ز����
double BP_Out[N][Out];  //BP�����������

FILE* fpstr; //�����ļ�ָ��

//�����й��
static void SetPositionByLine(FILE *fp, int nLine)
{
	int i = 0;
	char buffer[MAX_LENTH_OF_LINE];
    fseek(fp, 0, SEEK_SET);     //��λ����ͷ
	for (i = 0; i < nLine; i++) {
		fgets(buffer, MAX_LENTH_OF_LINE, fp);
    }
}

//���ı���ȡ���� �ı��� �׵�ַ ���� ����
static void getTxtData(char *FileName, double *Data, int Row, int Column)
{
    int i, j;
    FILE *fp;
	fp = fopen(FileName, "r");
    if(fp == NULL) {
        printf("��ȡʧ��\r\n");
        return;
    }    
    for(i = 0; i < Row; i++) {
        SetPositionByLine(fp, i); //�ı����ת����i��
        for(j = 0; j < Column; j++) {
        fscanf(fp, "%lf ", Data + i * Column + j); //������
        }
    }
    fclose(fp);
}

//�����������ı� �ı��� �׵�ַ ���� ����
static void saveTxtData(char *FileName, double *Data, int Row, int Column)
{
    int i, j;
    FILE *fp;
	fp = fopen(FileName, "w+");
    if(fp == NULL) {
        printf("����ʧ��\r\n");
        return;
    }    
    for(i = 0; i < Row; i++) {
        for(j = 0; j < Column; j++) {
        fprintf(fp, "%lf ", *(Data + i * Column + j)); //д����
        }
        fprintf(fp, "\n"); //����     
    }
    fclose(fp);
}

//������ʼ��
static void parm_init() 
{
    int i,j;
    srand((unsigned)time(NULL));    //���������������
    //��ʼ�� ����㵽���ز� Ȩֵ
    for(i = 0; i < In; i++) {
        for(j = 0; j < Hide; j++) {
            weight_IH[i][j] = (double)(rand() % 20 - 10) / 10; //ȡ�����-1.0 ~ +1.0
        }
    }
    //��ʼ�� ���ز� ƫ��  
    for(i = 0; i < Hide; i++) {
            bias_H[i] = (double)(rand() % 10) / 10; //ȡ�����0.0 ~ +1.0
    }
    //��ʼ�� ���ز㵽����� Ȩֵ
    for(i = 0; i < Hide; i++) {
        for(j = 0; j < Out; j++) {
            weight_HO[i][j] = (double)(rand() % 20 - 10) / 10; //ȡ�����-1.0 ~ +1.0
        }
    }
    //��ʼ�� ����� ƫ�� 
    for(i = 0; i < Out; i++) {
            bias_O[i] = (double)(rand() % 10) / 10; //ȡ�����0.0 ~ +1.0
    }
    for(i = 0; i < N; i++) {
        error[i] = 0; //��ʼ��һ����������� 
        for(j = 0; j < Hide; j++) {        
            delta_IH[i][j] = 0; //����㵽���ز� ����ֵ
        } 
        for(j = 0; j < Out; j++) {
            delta_HO[i][j] = 0; //���ز㵽��� ����ֵ
        }         
    }
    
}

//�����
static double sigmoid(double x)
{
    double tmp = 1.0 / (1.0 + exp(-x));
    return tmp;
}
 
//��ȡ���ڵ���� �����׵�ַ �������� Ȩֵ�׵�ַ Ȩֵ���� ƫ��
static double getOneNodeValue(double *value, int valueColumn, double *weight, int weightColumn, double bias)
{
    int i = 0 ;double tmp = 0.0;
    for (i = 0 ; i < valueColumn ; i++)
        tmp += *(value+i) * *(weight+i * weightColumn); //x1*w1 + x2*w2 + x3*w3 + ....
    tmp -= bias; //ƫ��
    return sigmoid(tmp);
}

//��ȡѵ������ �������
static void getTrainData()
{
    int i,j;
    //��ȡ�ı�����ȡ����
    getTxtData("trainDataIn.txt", *train_In, N, In);   //����ֵ
    getTxtData("trainDataOut.txt", *train_Out, N, Out);  //���ֵ
    for(i = 0; i < N; i++) {
        for(j = 0; j < In; j++) {
            //printf("%lf ", train_In[i][j]);     //��ӡ����
            train_In[i][j] /= In_divisor;       //������������������-1~1��       
        }    
        for(j = 0; j < Out; j++) {
            //printf("%lf ", train_Out[i][j]);     //��ӡ����
            train_Out[i][j] /= Out_divisor;      //�����������������-1~1��         
        }            
        //printf("\r\n");
     }
}

//ѵ��times��
static void train(int times)
{
    int i,j,k,n;
    for(n = 0; n < times; n++)
    {
        Err_Sum = 0;
        double delta_err[N] = {0};
        //������
        for(i  = 0; i < N; i++) {
            //�������ز���� in -> hide
            for(j = 0; j < Hide; j++) {
                hide_Out[i][j] = getOneNodeValue(train_In[i], In, *weight_IH+j, Hide, bias_H[j]);
            }
            //������������ hide -> out
            for(j = 0; j < Out; j++) {
                BP_Out[i][j] = getOneNodeValue(hide_Out[i], Hide, *weight_HO+j, Out, bias_O[j]);
                if((times - n) == 1) 
                printf("%lf \r\n",BP_Out[i][j]*Out_divisor); //���һ��ѵ����ӡÿ�����
            }
            //����������� 
            for(j = 0; j < Out; j++) {
                delta_err[i] = train_Out[i][j] - BP_Out[i][j]; //���
                error[i] += delta_err[i] * delta_err[i];  //���ƽ����
            }
            error[i] /= 2;    
            //�������ز�������Ȩֵ����ϵ��  �й�ʽ
            for(j = 0; j < Out; j++) {
                delta_HO[i][j] = delta_err[i] * BP_Out[i][j] * (1-BP_Out[i][j]);
            }
            //��������㵽���ز��Ȩֵ����ϵ��  �й�ʽ
            for(j = 0; j < Hide; j++) {               
                delta_IH[i][j] = 0; //��0
                for(k = 0; k < Out; k++) {
                    delta_IH[i][j] += delta_HO[i][k] * weight_HO[j][k] * hide_Out[i][j] * (1- hide_Out[i][j]);                       
                }  
            }
            Err_Sum += error[i]; //����
        }       
        //������
        //����weight_IH
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
        //����bias_H
        for(i = 0; i < Hide; i++) {
            temp = 0;
            for(j = 0; j < N; j++) {
                temp -= delta_IH[j][i];
            }
            bias_H[i] += rate * temp;
        }
        //����weight_HO
        for(i = 0; i < Hide; i++) {
            for(j = 0;j < Out; j++) {
                temp = 0;
                for (k = 0; k < N; k++) {
                    temp += delta_HO[k][j] * hide_Out[k][i];
                }
                weight_HO[i][j] += rate * temp;
            }
        }
        //����bias_O
        for (i = 0; i < Out; i++) {
            temp = 0;
            for(j = 0;j < N; j++) {
                temp -= delta_HO[j][i];
            }
            bias_O[i] += rate * temp;
        }
        if(Err_Sum < Err_max) break; //��������ֵ ����ѭ�� ֹͣѵ��
        if((n % (times / 10)) == 0) { //ÿTimes/10�δ�ӡһ����Ϣ
        printf("����:%d ���:%lf \r\n", n + times / 10, Err_Sum); 
        //printf("----------------------------------------------------------\r\n");
        }
    }
}

//����(������ / In_divisor)���������׵�ַ ��������׵�ַ
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
    parm_init(); //������ʼ��
    printf("�Ƿ��ȡ���� 1: ��ȡ 0: ����ȡ\r\n");
    scanf("%d", &mod);
    if(mod == 1) {
        //��ȡ����
        getTxtData("weight_IH.txt", *weight_IH, In, Hide);
        getTxtData("weight_HO.txt", *weight_HO, Hide, Out);
        getTxtData("bias_H.txt", bias_H, Hide, 1);
        getTxtData("bias_O.txt", bias_O, Out, 1);
    }

    printf("�Ƿ�ѵ�� 1: ѵ�� 0: ��ѵ��\r\n");
    scanf("%d", &mod);
    if(mod == 1) {
        getTrainData();    
        printf("ѵ��������\r\n");
        scanf("%d", &Times);    
        train(Times);        
        printf("-------------------------ѵ�����-------------------------\r\n" );        
        printf("\r\nѵ����� ���:%lf\r\n", Err_Sum);
    }

    getTxtData("testData.txt", *testData, Ntest, In); //��ȡ��������
    for(i = 0; i < In; i++) {
        for(j = 0; j < Ntest; j++) {
         testData[j][i] /= In_divisor;  //��������
        } 
    }

    printf("\r\n--------------------------���Լ�--------------------------\r\n" );
    for(i = 0; i < Ntest; i++) { 
        test(testData[i],OutData[i]);
        for(j = 0; j < Out; j++) {
            printf("%lf ", OutData[i][j]);                
        }
        printf("\r\n");   
    }
    //�洢����
    saveTxtData("weight_IH.txt", *weight_IH, In, Hide);
    saveTxtData("weight_HO.txt", *weight_HO, Hide, Out);
    saveTxtData("bias_H.txt", bias_H, Hide, 1);
    saveTxtData("bias_O.txt", bias_O, Out, 1);
          
    return 0;
}
