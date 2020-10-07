from mpi4py import MPI
from cobaya.run import run
from cobaya.log import LoggedError
import scipy.stats as st
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

error_type = None  # "A"|"B"|"C"|None

n_evals = 0

def loglikelihood(x):
    global n_evals
    n_evals += 1
    # ERROR A: likelihood fails at evaluation in one process
    if error_type == "A" and rank == 0 and n_evals == 50:
        raise ValueError("Likelihood failed!")
    # ERROR B: chains waiting too long for each other
    if error_type == "B" and rank == 0 and n_evals >= 50:
        sleep(0.5)
    return st.norm.logpdf(x)

info = {
    "params": {"x": {"prior": {"min": -5, "max": 5}, "proposal": 0.5}},
    "likelihood": {"gaussian": loglikelihood},
    "sampler": {"mcmc": None}}

# ERROR C: chain stuck!
if error_type == "C" and rank == 0:
    info["params"]["x"]["proposal"] *= 100


success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    pass
# Di it work? (e.g. did not get stuck)
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")


# Now gather chains of all MPI processes in rank 0

copy_and_skip_1st_3rd = lambda chain: chain[int(len(chain) / 3):]

all_chains = comm.gather(mcmc.products()["sample"], root=0)
# Now (optionally) concatenate them in rank = 0
# Skip 1st 3rd of each chain
if rank == 0:
    print([len(c) for c in all_chains])
    full_chain = copy_and_skip_1st_3rd(all_chains[0])
    for chain in all_chains[1:]:
        full_chain.append(copy_and_skip_1st_3rd(chain))
    print(len(full_chain))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(full_chain["x"], weights=full_chain["weight"])
    plt.show()

