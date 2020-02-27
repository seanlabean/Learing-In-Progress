import pickle

#sim_time = quantity<133947058727.0 s>
sim_time = 133947058728.0
fake_torch_loop = {'dt': sim_time, 'it': 11}

pickle.dump(fake_torch_loop, open("fake_torch_loop0001.pickle", "wb"))
