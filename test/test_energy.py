from energy import Funnel, ManyWell, GMM25, GaussianEnergy, UniformEnergy

import torch


def test_funnel_sample_device_check():
    cuda_energy_function = Funnel("cuda:0", 10, 3)
    sample = cuda_energy_function.sample(10)
    assert sample.device == torch.device("cuda:0")

    cpu_energy_function = Funnel("cpu", 10, 3)
    sample = cpu_energy_function.sample(10)
    assert sample.device == torch.device("cpu")


# energy method use log_prob, so we only test log_prob.
def test_funnel_log_prob_device_check():
    energy_function = Funnel("cuda:0", 10, 3)
    e = energy_function.log_prob(torch.zeros(2, 10, device="cpu"))
    assert e.device == torch.device("cpu")

    energy_function = Funnel("cpu", 10, 3)
    e = energy_function.log_prob(torch.zeros(2, 10, device="cuda:0"))
    assert e.device == torch.device("cuda:0")


def test_manywell_sample_device_check():
    energy_function = ManyWell(device="cuda:0")
    sample = energy_function.sample(1000)
    assert sample.device == torch.device("cuda:0")


def test_manywell_log_prob_device_check():
    energy_function = ManyWell(device="cuda:0")
    e = energy_function.log_prob(torch.zeros(2, 32, device="cpu"))
    assert e.device == torch.device("cpu")

    energy_function = ManyWell(device="cpu")
    e = energy_function.log_prob(torch.zeros(2, 32, device="cuda:0"))
    assert e.device == torch.device("cuda:0")


def test_manywell_log_prob_shape_check():
    energy_function = ManyWell(device="cpu")
    e = energy_function.log_prob(torch.zeros(32, 2, 32, device="cpu"))

    assert e.shape == (32, 2)


def test_gmm25_log_prob_device_check():
    energy_function = GMM25(device="cuda:0", dim=2)
    e = energy_function.log_prob(torch.zeros(10, 2, device="cpu"))
    assert e.device == torch.device("cpu")


def test_gmm25_log_prob_shape_check():
    energy_function = GMM25(device="cpu", dim=2)
    e = energy_function.log_prob(torch.zeros(100, 32, 2, 2, device="cpu"))

    assert e.shape == (100, 32, 2)


def test_gaussian_energy_device_check():
    energy_function = GaussianEnergy("cuda:0", 10, 1.0)
    e = energy_function.energy(torch.zeros(10, 10, device="cpu"))
    assert e.device == torch.device("cpu")


def test_gaussian_energy_shape_check():
    energy_function = GaussianEnergy("cuda:0", 10, 1.0)
    e = energy_function.energy(torch.zeros(10, 32, 10, device="cpu"))
    assert e.shape == (10, 32)


def test_uniform_energy_device_check():
    energy_function = UniformEnergy("cuda:0", 10, 30)
    e = energy_function.energy(torch.zeros(10, 10, device="cpu"))
    assert e.device == torch.device("cpu")


def test_uniform_energy_shape_check():
    energy_function = UniformEnergy("cuda:0", 10, 30)
    e = energy_function.energy(torch.zeros(10, 32, 10, device="cpu"))
    assert e.shape == (10, 32)
