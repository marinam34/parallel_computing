#pragma once

struct Timer {
    double total_time = 0.0;
    double init_time = 0.0;
    double compute_time = 0.0;
    double copy_h2d_time = 0.0;
    double copy_d2h_time = 0.0;
    
    double comm_pack_time = 0.0;
    double comm_send_time = 0.0;
    double comm_recv_time = 0.0;
    double comm_wait_time = 0.0;
    double comm_unpack_time = 0.0;
    double comm_total_time = 0.0; 

    double finalize_time = 0.0;
};
