import numpy as np

# Extrinsic parameters cameras 1,5,13,21
Rot_1 = [-8.9178619368380785e-03, -9.9947025135471135e-01, 3.1299974367819061e-02, -3.2694659041264756e-01,
         -2.6666499697833967e-02, -9.4466651513188005e-01,
         9.4500074008226076e-01, -1.8657825457862154e-02,
         -3.2653558273665029e-01]
T_1 = [1.3538821883123289e-01, 8.5448637042441988e-01,
       3.3707566630661425e+00]

Rot_5 = [8.8295905870293179e-01, -4.6944958495306982e-01,
         6.2276947891684586e-04, -1.6882182749933156e-01,
         -3.1876395303440941e-01, -9.3267825792481251e-01,
         4.3804393753848403e-01, 8.2341157960844069e-01,
         -3.6070885677026748e-01]
T_5 = [-4.0983427328061879e-02, 9.2175130735523203e-01,
       3.2572920695554815e+00]

Rot_13 = [3.3986409410032725e-02, 9.9466922811912029e-01,
          -9.7355280330994998e-02, 3.7627065128700110e-01,
          -1.0297878034649266e-01, -9.2076911751991575e-01,
          -9.2588623542823512e-01, -5.3382985362254720e-03,
          -3.7776471726892213e-01]
T_13 = [-2.5072212183723241e-01, 7.1782595010966588e-01,
        3.7906980375345154e+00]

Rot_21 = [-7.6391076314527662e-01, -6.4318807524924493e-01,
          -5.2435158128613768e-02, -2.0606952193746192e-01,
          3.2013182793730210e-01, -9.2468965868013664e-01,
          6.1153552478925288e-01, -6.9557509486660984e-01,
          -3.7709360816916493e-01]
T_21 = [3.6371147388865571e-01, 6.7672702638064697e-01,
        3.6950499159012988e+00]

# Intrinsic parameters cameras 1,5,13,21

K_1 = [1.1562489404251594e+03, 0., 5.1707325922624091e+02, 0.,
       1.1562521516982895e+03, 5.1809786731690701e+02, 0., 0., 1.]
dist_1 = [-2.2387073221830378e-01, 3.2215956543532548e-01,
          -8.4812307921968452e-04, -5.5963210135421984e-04,
          -3.9610266919774328e-01]

K_5 = [1.1526687308608459e+03, 0., 5.1431412391330969e+02, 0.,
       1.1524155212596120e+03, 5.1628851722352044e+02, 0., 0., 1.]
dist_5 = [-2.1910350691100472e-01, 2.6237879258319846e-01,
          -8.9223322720105548e-04, 6.1981923776207844e-04,
          -2.2592111204690682e-01]

K_13 = [1.1529498055793485e+03, 0., 4.9298920954487761e+02, 0.,
        1.1532986295467629e+03, 5.0179242775799321e+02, 0., 0., 1.]
dist_13 = [-2.2828828675807583e-01, 4.2594778824369572e-01,
           -5.3203685481689815e-05, -4.8701376406823762e-04,
           -8.3331835163731871e-01]

K_21 = [1.1540747802703579e+03, 0., 5.2914503340923477e+02, 0.,
        1.1540132992768401e+03, 5.3183873356175525e+02, 0., 0., 1.]
dist_21 = [-2.0679357917310773e-01, 2.0517033102360435e-01,
           -1.3871078904436273e-03, 2.5016397706686212e-04,
           -1.1508540030110884e-01]


def get_all_camera_parameters():
    rotational_matrices = [np.reshape(Rot_1, (3, 3)),
                           np.reshape(Rot_5, (3, 3)),
                           np.reshape(Rot_13, (3, 3)),
                           np.reshape(Rot_21, (3, 3))]
    translation_vectors = [np.reshape(T_1, (1, 3)),
                           np.reshape(T_5, (1, 3)),
                           np.reshape(T_13, (1, 3)),
                           np.reshape(T_21, (1, 3))]
    camera_matrices = [np.reshape(K_1, (3, 3)),
                       np.reshape(K_5, (3, 3)),
                       np.reshape(K_13, (3, 3)),
                       np.reshape(K_21, (3, 3))]
    camera_distortions = [np.reshape(dist_1, (1, 5)),
                          np.reshape(dist_5, (1, 5)),
                          np.reshape(dist_13, (1, 5)),
                          np.reshape(dist_21, (1, 5))]

    return rotational_matrices, translation_vectors, camera_matrices, camera_distortions