# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:07:14 2023

@author: sagar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, draw
import  glob
import magfield.magmodule_sagar as mms

#################################################
"""      Give the Pump probe image path   """
#################################################
files_Pu_Pr_DP = sorted(glob.glob("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Magnet\\P D\\pump probe\\*.tif"))
files_Pu_Pr_QT = sorted(glob.glob("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Magnet\\T Q\\pump probe\\*.tif"))


fileset = [3,2,2,2,3,3,3,3,3,2,2,2,2,2,2]
retro = np.round(np.linspace(26.5,27.9,15),2)

index = 0
successful_files = 0
bg_factor_array = []

#################################################
"""  BS + Filter factors at experiment time  """
#################################################
P_filter = 792.82
D_filter = 415.2
T_filter = 771.59
Q_filter = 333.33



#################################################
"""      Give the Pump only image """
#################################################
pump_only_DP = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Magnet\\P D\\pump only\\P_006.tif")
pump_only_QT = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Magnet\\T Q\\pump only\\C_006.tif")

bg_fac_pump_only_DP = 1.01
bg_fac_pump_only_QT = 1.01

#################################################

tol = 0.03


#################################################
#################################################
#################################################
#################################################



for i in range(len(fileset)):
    
    Filename=[]
    Retro=[]
    P=[]
    D=[]
    T=[]
    Q=[]
    S1=[]
    S2=[]
    S3=[]
    modS=[]
    BG_Fac=[]
    Ellip=[]
    Phi=[]
    
    for j in range(fileset[i]):
        #################################################
        """     Give the pump-probe image  """
        #################################################
        Pu_Pr_DP = io.imread(files_Pu_Pr_DP[index])
        Pu_Pr_QT = io.imread(files_Pu_Pr_QT[index])
    
        pump_only_D , pump_only_P = mms.get_image_tl_and_br(image = pump_only_DP, background_factor = bg_fac_pump_only_DP)
        pump_only_Q , pump_only_T = mms.get_image_tl_and_br(image = pump_only_QT, background_factor = bg_fac_pump_only_QT)
        
        D_noise = mms.get_nonzero_value_average(pump_only_D)
        P_noise = mms.get_nonzero_value_average(pump_only_P)
        Q_noise = mms.get_nonzero_value_average(pump_only_Q)
        T_noise = mms.get_nonzero_value_average(pump_only_T)
        
        try:
            P_int, D_int, Q_int, T_int, s, mod_s, background_factor = mms.stokes_generator( image_P = Pu_Pr_DP,
                                                                                            image_C = Pu_Pr_QT,
                                                                                            p_filter = P_filter,
                                                                                            d_filter = D_filter,
                                                                                            t_filter = T_filter,
                                                                                            q_filter = Q_filter,
                                                                                            d_image_pump_noise = D_noise,
                                                                                            p_image_pump_noise = P_noise,
                                                                                            t_image_pump_noise = T_noise,
                                                                                            q_image_pump_noise = Q_noise,
                                                                                            tol = tol)
            print(f"file: {files_Pu_Pr_DP[index][7:]}")
            print(f"s:  {s}")
            print(f"mods: {mod_s}")
            print(f"P signal integrated: {P_int}")
            print(f"D signal integrated: {D_int}")
            print(f"T signal integrated: {T_int}")
            print(f"Q signal integrated: {Q_int}")
            print(f"background factor: {background_factor}")
            print("\n#########################################")
            
            s1 = s[0,0]
            s2 = s[1,0]
            s3 = s[2,0]
            
            ellipticity = 0.5*np.arctan(s3/np.sqrt(s1**2+s2**2))
            phi = 0.5*np.arctan(s2/s1)
            
            Retro.append(retro[i])
            Filename.append(f"file:  {files_Pu_Pr_DP[index][-7:]}")
            P.append(P_int)
            D.append(D_int)
            Q.append(Q_int)
            T.append(T_int)
            S1.append(s1)
            S2.append(s2)
            S3.append(s3)
            modS.append(mod_s)
            BG_Fac.append(background_factor)
            Ellip.append(ellipticity)
            Phi.append(phi)
        
            successful_files += 1
            bg_factor_array.append(background_factor)
            
        except:
            None
            
        index += 1
        
    print(f"Filename: {Filename} \n\nRetro {Retro} \n\nP {P} \n\nD {D} \n\nQ {Q} \n\nT {T} \n\ns1 {S1} \n\ns2 {S2} \n\ns3 {S3} \n\nmod_s {modS} \n\nBg fac{BG_Fac} \n\nEllip {Ellip} \n\nPhi {Phi}")
    data = {'File': Filename,
            'Retro': Retro,
            'P': P,
            'D': D,
            'T': T,
            'Q': Q,
            's1': S1,
            's2': S2,
            's3': S3,
            'ModS': modS,
            'Bg factor': BG_Fac,
            'ellipticity': Ellip,
            'faraday rot': Phi
            }

    df = pd.DataFrame(data)
    
    df.to_csv(f'D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Magnet\\csv files\\{retro[i]}.csv', index=False)
   
    

print(f"successful_files:  {successful_files}")

plt.plot(bg_factor_array)
#plt.ylim(0.75,2)
plt.xlabel("Files")
plt.ylabel("Background factor")
plt.show()