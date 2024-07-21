from energy import Funnel, ManyWell

import torch


def test_funnel_sample():
    cuda_energy_function = Funnel("cuda:0", 10, 3)
    sample = cuda_energy_function.sample(10)
    assert sample.device == torch.device("cuda:0")

    cpu_energy_function = Funnel("cpu", 10, 3)
    sample = cpu_energy_function.sample(10)
    assert sample.device == torch.device("cpu")


def test_funnel_log_prob():
    energy_function = Funnel("cuda:0", 10, 3)
    energy_function.log_prob(torch.zeros(2, 10, device="cuda:0"))

    try:
        energy_function = Funnel("cpu", 10, 3)
        energy_function.log_prob(torch.zeros(2, 10, device="cuda:0"))
    except:
        return

    raise Exception("Exception must be raised by device conflict")


def test_manywell_sample():
    energy_function = ManyWell(device="cuda:0")
    sample = energy_function.sample(1000)
