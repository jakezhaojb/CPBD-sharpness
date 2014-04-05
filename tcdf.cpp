#include "tcdf.h"
#define MAXIT 100
#define EPS 3.0e-7f
#define FPMIN 1.0e-30f

float tcdf(float t, unsigned int df)
{
  float x=(t+sqrt(t*t+df))/(2*sqrt(t*t+df));
  return betai(df/2.0f,df/2.0f,x);
}

float betai(float a, float b, float x)
{
  float bt;
  
  if (x < 0.0 || x > 1.0) 
    nrerror("Bad x in routine betai");
  if (x == 0.0 || x == 1.0) 
    bt=0.0f;
  else
    bt=exp(gammln(a+b)-gammln(a)-gammln(b)+a*log(x)+b*log(1.0f-x));
  if (x < (a+1.0)/(a+b+2.0))
    return bt*betacf(a,b,x)/a;
  else
    return 1.0f-bt*betacf(b,a,1.0f-x)/b;
}

float gammln(float xx)
{
  double x,y,tmp,ser;
  static double cof[6]={76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
  int j;
  y=x=xx;
  tmp=x+5.5;
  tmp-=(x+0.5)*log(tmp);
  ser=1.000000000190015;
  for (j=0; j<=5; j++)
    ser+= cof[j]/++y;
  return static_cast<float>( -tmp+log(2.5066282746310005*ser/x) );
}

float betacf(float a, float b, float x)
{
	int m, m2;
	float aa, c, d, del, h, qab, qam, qap;

	qab=a+b;
	qap=a+1.0f;
	qam=a-1.0f;
	c=1.0f;
	d=1.0f-qab*x/qap;
	if (fabs(d) < FPMIN) d=FPMIN;
	d=1.0f/d;
	h=d;
	for (m=1; m<MAXIT; m++)
	{
		m2=2*m;
		aa=m*(b-m)*x/((qam+m2)*(a+m2));
		d=1.0f+aa*d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=1.0f+aa/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0f/d;
		h*=d*c;
		aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
		d=1.0f+aa*d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=1.0f+aa/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0f/d;
		del=d*c;
		h*=del;
		if (fabs(del-1.0f) < EPS) break;
	}
	if (m>MAXIT)
		nrerror("a or b too big, or MAXIT too small in betacf");

	return h;
}

void nrerror(const char* error_text){
  fprintf(stderr, "%s\n", error_text);
  exit(1);
}


int main(int argc, const char *argv[])
{
  printf("Testing: %f\n", tcdf(1.23, 4));
  printf("Testing: %f\n", tcdf(5.67, 10));
  return 0;
}
