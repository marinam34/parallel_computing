#include "config.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static bool eq(const char* a, const char* b){ return std::strcmp(a,b)==0; }

Config Config::parse_cmd(int argc, char** argv){
    Config c;
    for(int i=1;i<argc;i++){
        if(eq(argv[i],"--L") && i+1<argc){
            std::string s = argv[++i];
            if(s=="pi" || s=="PI") c.Lx=c.Ly=c.Lz=M_PI;
            else c.Lx=c.Ly=c.Lz=std::stod(s);
        } else if(eq(argv[i],"--Lx") && i+1<argc){ c.Lx=std::stod(argv[++i]); }
        else if(eq(argv[i],"--Ly") && i+1<argc){ c.Ly=std::stod(argv[++i]); }
        else if(eq(argv[i],"--Lz") && i+1<argc){ c.Lz=std::stod(argv[++i]); }
        else if(eq(argv[i],"--N")  && i+1<argc){ c.Nx=c.Ny=c.Nz=std::stoi(argv[++i]); }
        else if(eq(argv[i],"--Nx") && i+1<argc){ c.Nx=std::stoi(argv[++i]); }
        else if(eq(argv[i],"--Ny") && i+1<argc){ c.Ny=std::stoi(argv[++i]); }
        else if(eq(argv[i],"--Nz") && i+1<argc){ c.Nz=std::stoi(argv[++i]); }
        else if(eq(argv[i],"--T")  && i+1<argc){ c.T = std::stod(argv[++i]); }
        else if(eq(argv[i],"--cfl")&& i+1<argc){ c.cfl = std::stod(argv[++i]); }
        else if(eq(argv[i],"--help")){
            std::cout<<"Args:\n--L <1|pi> or --Lx --Ly --Lz\n--N <int> or --Nx --Ny --Nz\n--T <final time>\n--cfl <0..1>\n";
            std::exit(0);
        }
    }
    c.hx = c.Lx / c.Nx;
    c.hy = c.Ly / c.Ny;
    c.Nz_internal = c.Nz - 2;
    if(c.Nz_internal <= 0) throw std::runtime_error("Nz must be >= 3");
    c.hz = c.Lz / (c.Nz - 1);
    c.omega = M_PI;
    const double kx = 2.0*M_PI/c.Lx;
    const double ky = 2.0*M_PI/c.Ly;
    const double kz = 1.0*M_PI/c.Lz;
    const double kabs = std::sqrt(kx*kx + ky*ky + kz*kz);
    c.a = c.omega / kabs;
    const double hxy = (c.hx < c.hy ? c.hx : c.hy);
    const double hmin = (hxy < c.hz ? hxy : c.hz);
    c.dt = c.cfl * hmin / (c.a * std::sqrt(3.0));
    return c;
}

