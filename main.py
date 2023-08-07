from model_uplink import Model
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

antenna_num = 8
metasurface_len = 31


model = Model(
    antenna_num=antenna_num,
    antenna_space=0.58,
    metasurface_len=metasurface_len,
    antenna_metasurface_distance=0.02,
    frequency=30e9,
    start_angle=-60,
    end_angle=60,
    angle_num=13,
    w1=2,
    w2=2,
)

paras = model.parameters()
optimizer = optim.Adam(paras, lr=0.001)

bar = tqdm(range(10000))

loss_list = []
for i in bar:
    loss = model()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_list.append(loss.detach().numpy())
    bar.set_description_str("loss: {:.4f}".format(loss.item()))

fig = plt.figure()
plt.plot(loss_list)
plt.show()
# plt.savefig("./out/loss.png")

np.save("./setting_output/out_mmwave/metasurface_theta.npy", model.metasurface_theta.detach().cpu().numpy())
np.save("./setting_output/out_mmwave/antenna_theta.npy", model.antenna_theta.detach().cpu().numpy())
np.save("./setting_output/out_mmwave/air_spread.npy", model.air_spread.detach().cpu().numpy())
np.save("./setting_output/out_mmwave/antenna_A.npy", model.antenna_A.detach().cpu().numpy())
np.save("./setting_output/out_mmwave/metasurface_A.npy", model.metasurface_A.detach().cpu().numpy())
np.save("./setting_output/out_mmwave/show_spread.npy", model.show_spread.detach().cpu().numpy())


metasurface_opti = model.metasurface_theta.detach().cpu().numpy()
phase_map = metasurface_opti.reshape(metasurface_len, metasurface_len)
# np.save("./results/metasurface_uplink.npy", phase_map)
phase_map = np.unwrap(phase_map)
plt.figure()
plt.imshow(phase_map)
plt.colorbar()
plt.show()
# plt.savefig("./results/metasurface_uplink.png")     
