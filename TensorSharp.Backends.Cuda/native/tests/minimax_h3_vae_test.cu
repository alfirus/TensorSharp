#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../kernels/tensorsharp_minimax_h3.cuh"

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Buffer {
    float * p = nullptr;
    explicit Buffer(const std::vector<float> & values) {
        check(cudaMalloc(&p, values.size()*sizeof(float)));
        check(cudaMemcpy(p, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
    explicit Buffer(size_t count) { check(cudaMalloc(&p, count*sizeof(float))); }
    ~Buffer() { if (p) cudaFree(p); }
};
static void compare(const std::string & name, Buffer & output, const std::vector<double> & expected, double tolerance) {
    check(cudaDeviceSynchronize());
    std::vector<float> actual(expected.size());
    check(cudaMemcpy(actual.data(), output.p, actual.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double maximum = 0;
    for (size_t i=0;i<actual.size();i++) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(name+" nonfinite output");
        maximum = std::max(maximum, std::abs(actual[i]-expected[i]));
    }
    std::cout << name << " values=" << actual.size() << " max_abs_error=" << maximum << '\n';
    if (maximum > tolerance) throw std::runtime_error(name+" exceeded its independent oracle bound");
}
static std::vector<float> signal(size_t count, float scale=1) {
    std::vector<float> values(count);
    for (size_t i=0;i<count;i++) values[i]=std::sin(float(i)*.37f)*scale;
    return values;
}
static void transform(Buffer & out, Buffer & input, Buffer * a, Buffer * b, int count, int op,
        int p0, int p1=0, int p2=0, int p3=0, int p4=0, int p5=0, int p6=0, int p7=0, int p8=0) {
    ts_h3_transform_f32<<<(count+255)/256,256>>>(input.p,a ? a->p : nullptr,b ? b->p : nullptr,
        out.p,count,op,p0,p1,p2,p3,p4,p5,p6,p7,p8,0);
    check(cudaGetLastError());
}
static void reflect() {
    constexpr int h=4,w=5,c=3,k=3,y0=2,rows=2,count=rows*w*c*k*k;
    auto x=signal(h*w*c); Buffer input(x),output(count); std::vector<double> expected(count);
    for(int r=0;r<rows*w;r++) for(int ch=0;ch<c;ch++) for(int ky=0;ky<k;ky++) for(int kx=0;kx<k;kx++) {
        int sy=y0+r/w-1+ky,sx=r%w-1+kx;
        sy=sy<0 ? -sy : (sy>=h ? 2*(h-1)-sy : sy);
        sx=sx<0 ? -sx : (sx>=w ? 2*(w-1)-sx : sx);
        expected[r*c*k*k+(ch*k+ky)*k+kx]=x[(sy*w+sx)*c+ch];
    }
    transform(output,input,nullptr,nullptr,count,0,c,h,w,k,w,1,1,1,y0);
    compare("reflect-im2col-band-corners",output,expected,0);
}
static void columns(bool transpose) {
    constexpr int time=11,c=3,k=5,rows=9,start=2,count=rows*c*k;
    auto x=signal(time*c); Buffer input(x),output(count); std::vector<double> expected(count);
    for(int row=0;row<rows;row++) for(int ch=0;ch<c;ch++) for(int tap=0;tap<k;tap++) {
        int source=transpose ? row+start-tap : (row+start)*2-3+tap*2;
        bool valid=!transpose || source%3 == 0;
        if(transpose)source/=3;
        expected[(row*c+ch)*k+tap]=valid && source>=0 && source<time ? x[source*c+ch] : 0;
    }
    transform(output,input,nullptr,nullptr,count,1,c,time,k,transpose ? 3 : 2,3,2,start,transpose);
    compare(transpose ? "transpose-im2col-overlap" : "strided-dilated-im2col",output,expected,0);
}
static void resample(bool up) {
    constexpr int time=9,c=2,k=12;
    const int rows=up ? 18 : 5,count=rows*c;
    auto x=signal(time*c), filter=signal(k,.13f); Buffer input(x),a(filter),output(count);
    std::vector<double> expected(count);
    for(int t=0;t<rows;t++) for(int ch=0;ch<c;ch++) {
        double sum=0;
        for(int tap=0;tap<k;tap++) {
            int source=up ? t+15-11+tap : t*2-5+tap;
            if(up) {
                if(source<0 || source >= (time+10)*2 || source%2 != 0)continue;
                source=source/2-5;
            }
            source=std::max(0,std::min(time-1,source));
            sum+=double(filter[tap])*x[source*c+ch];
        }
        expected[t*c+ch]=sum;
    }
    transform(output,input,&a,nullptr,count,2,c,time,k,up ? 1 : 2,up ? 11 : 5,up,up ? 5 : 0,up ? 15 : 0);
    compare(up ? "zero-stuff-replicate-up-filter" : "replicate-down-filter",output,expected,1e-6);
}
static void snake() {
    constexpr int c=2,time=17,count=c*time;
    auto x=signal(count,4);std::vector<float> alpha{.7f,1.3f},inv{1/1.1f,1/.8f};
    Buffer input(x),a(alpha),b(inv);std::vector<double> expected(count);
    for(int i=0;i<count;i++) {
        double sine=std::sin(double(alpha[i%c])*x[i]);
        expected[i]=x[i]+sine*sine*inv[i%c];
    }
    transform(input,input,&a,&b,count,3,c);
    compare("snakebeta-in-place",input,expected,2e-6);
}
static void pool() {
    constexpr int heads=4,dim=32,lat=4,time=11,count=time*lat;
    auto x=signal(time*heads*dim);Buffer input(x),output(count);std::vector<double> expected(count);
    for(int t=0;t<time;t++) for(int ch=0;ch<lat;ch++)
        for(int h=0;h<heads;h++) for(int d=0;d<dim/lat;d++)
            expected[t*lat+ch]+=x[(t*heads+h)*dim+ch*(dim/lat)+d]/double(heads*(dim/lat));
    transform(output,input,nullptr,nullptr,count,4,heads,dim,lat);
    compare("attention-head-adaptive-average-pool",output,expected,1e-6);
}
static void groupnorm() {
    constexpr int frames=2,rows=19*17,c=8,groups=4,width=c/groups,count=frames*rows*c;
    auto x=signal(count,80000);std::vector<float> gain(c),bias(c);
    for(int ch=0;ch<c;ch++){gain[ch]=.6f+ch*.03f;bias[ch]=-.2f+ch*.07f;}
    for(int i=rows*c;i<count;i++)x[i]+=12000;
    Buffer input(x),a(gain),b(bias);std::vector<double> expected(count);
    for(int f=0;f<frames;f++)for(int g=0;g<groups;g++) {
        double sum=0,variance=0;
        for(int r=0;r<rows;r++)for(int ch=0;ch<width;ch++)sum+=x[(f*rows+r)*c+g*width+ch];
        float mean=float(sum/(rows*width));
        for(int r=0;r<rows;r++)for(int ch=0;ch<width;ch++) {
            float v=x[(f*rows+r)*c+g*width+ch]-mean;variance+=double(v)*v;
        }
        double scale=1/std::sqrt(variance/(rows*width)+1e-6);
        for(int r=0;r<rows;r++)for(int ch=0;ch<width;ch++) {
            int channel=g*width+ch,index=(f*rows+r)*c+channel;
            double v=(x[index]-mean)*scale*gain[channel]+bias[channel];
            expected[index]=v/(1+std::exp(-v));
        }
    }
    ts_h3_groupnorm_silu_f32<<<frames*groups,256>>>(input.p,a.p,b.p,input.p,rows,c,groups,1e-6f);
    check(cudaGetLastError());compare("centered-double-frame-groupnorm-in-place",input,expected,2e-5);
}
int main() {
    try {
        int count=0;check(cudaGetDeviceCount(&count));
        if(count==0){std::cerr<<"CUDA device unavailable\n";return 77;}
        cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0));
        std::cout<<"device="<<device.name<<" compute="<<device.major<<'.'<<device.minor<<'\n';
        reflect();columns(false);columns(true);resample(true);resample(false);snake();pool();groupnorm();
        std::cout<<"PASS: 8 independent CUDA H3 VAE primitive oracle cases\n";
        return 0;
    } catch(const std::exception & ex) { std::cerr<<"FAIL: "<<ex.what()<<'\n';return 1; }
}
