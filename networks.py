# networks.py
import pandapower as pp
import pandapower.networks as pn

def create_campus_microgrid_network():
    net = pp.create_empty_network(sn_mva=10.)
    line_data = {"r_ohm_per_km": 0.15, "x_ohm_per_km": 0.08, "c_nf_per_km": 10.0,
                 "max_i_ka": 0.25, "type": "ol"}
    pp.create_std_type(net, line_data, name="campus_138_overhead", element="line")
    bus_feeder = pp.create_bus(net, vn_kv=13.8, name="Xcel Feeder")
    bus_tunnel1 = pp.create_bus(net, vn_kv=13.8, name="Utility Tunnel Bus 1")
    bus_tunnel2 = pp.create_bus(net, vn_kv=13.8, name="Utility Tunnel Bus 2")
    bus_tunnel3 = pp.create_bus(net, vn_kv=13.8, name="Utility Tunnel Bus 3")
    bus_steam_vault = pp.create_bus(net, vn_kv=0.48, name="Steam Vault Bus")
    bus_anderson_arena = pp.create_bus(net, vn_kv=0.48, name="Anderson Arena Bus")
    bus_fdc_vault = pp.create_bus(net, vn_kv=0.48, name="FDC Vault Bus")
    bus_parking_vault = pp.create_bus(net, vn_kv=0.48, name="Parking Vault Bus")
    bus_ow_oss_vault = pp.create_bus(net, vn_kv=0.48, name="OWS/OSS Vault Bus")
    bus_fac_design_vault = pp.create_bus(net, vn_kv=0.48, name="Facilities & Design Vault Bus")
    bus_microgrid_exp = pp.create_bus(net, vn_kv=0.48, name="Microgrid Expansion Bus")
    pp.create_transformer(net, bus_tunnel1, bus_steam_vault,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to Steam Vault")
    pp.create_transformer(net, bus_tunnel1, bus_anderson_arena,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to Anderson Arena")
    pp.create_transformer(net, bus_tunnel2, bus_fdc_vault,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to FDC Vault")
    pp.create_transformer(net, bus_tunnel2, bus_parking_vault,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to Parking Vault")
    pp.create_transformer(net, bus_tunnel3, bus_fac_design_vault,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to Facilities & Design")
    pp.create_transformer(net, bus_tunnel3, bus_ow_oss_vault,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to OWS/OSS")
    pp.create_transformer(net, bus_tunnel2, bus_microgrid_exp,
                          std_type="0.63 MVA 20/0.4 kV", name="TX to Microgrid Expansion")
    pp.create_ext_grid(net, bus_feeder, vm_pu=1.0, name="Xcel External Grid")
    pp.create_line(net, bus_feeder, bus_tunnel1,
                   length_km=0.3, std_type="campus_138_overhead", name="Feeder -> Tunnel1")
    pp.create_line(net, bus_tunnel1, bus_tunnel2,
                   length_km=0.5, std_type="campus_138_overhead", name="Tunnel1 -> Tunnel2")
    pp.create_line(net, bus_tunnel2, bus_tunnel3,
                   length_km=0.4, std_type="campus_138_overhead", name="Tunnel2 -> Tunnel3")
    pp.create_line(net, bus_tunnel3, bus_feeder,
                   length_km=0.7, std_type="campus_138_overhead", name="Tunnel3 -> Feeder")
    pp.create_load(net, bus=bus_steam_vault, p_mw=0.4, q_mvar=0.08, name="Steam Vault Load")
    pp.create_load(net, bus=bus_anderson_arena, p_mw=0.5, q_mvar=0.1, name="Anderson Arena Load")
    pp.create_load(net, bus=bus_fdc_vault, p_mw=0.2, q_mvar=0.04, name="FDC Vault Load")
    pp.create_load(net, bus=bus_parking_vault, p_mw=0.3, q_mvar=0.06, name="Parking Facility Load")
    pp.create_load(net, bus=bus_fac_design_vault, p_mw=0.25, q_mvar=0.05, name="Facilities & Design Load")
    pp.create_load(net, bus=bus_ow_oss_vault, p_mw=0.35, q_mvar=0.07, name="OWS/OSS Vault Load")
    pp.create_load(net, bus=bus_microgrid_exp, p_mw=0.4, q_mvar=0.08, name="Microgrid Expansion Load")
    pp.create_sgen(net, bus=bus_steam_vault, p_mw=0.3, q_mvar=0.0, name="Steam Vault Gen")
    pp.create_sgen(net, bus=bus_microgrid_exp, p_mw=0.2, q_mvar=0.0, name="Microgrid PV 1")
    pp.create_sgen(net, bus=bus_microgrid_exp, p_mw=0.2, q_mvar=0.0, name="Microgrid PV 2")
    return net

def create_ieee30_network():
    return pn.case_ieee30()

def create_ieee33bw_network():
    return pn.case33bw()

def create_cigre_mv_network():
    return pn.create_cigre_network_mv(with_der="all")
