from pathlib import Path
import csv

import pytest

from axcer.process import Axcer

TESTS_DIR = Path(__file__).parent
METRICS_FILE = TESTS_DIR / "test.csv"


@pytest.fixture
def axcer_instance() -> Axcer:

    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    return Axcer(metrics_file_path=METRICS_FILE)


def test_compress_prompt_and_metrics(axcer_instance: Axcer) -> None:
    """
    1. Compress 4 prompts and verify compressed output.
    2. Read the last 4 rows of tests/test.csv and verify metrics.
    """

    test_cases = [
        (
        # scitldr
            """Residual and skip connections play an important role in many current generative models. Although their theoretical and numerical advantages are understood, their role in speech enhancement systems has not been investigated so far. When performing spectral speech enhancement, residual connections are very similar in nature to spectral subtraction, which is the one of the most commonly employed speech enhancement approaches. Highway networks, on the other hand, can be seen as a combination of spectral masking and spectral subtraction. However, when using deep neural networks, such operations would normally happen in a transformed spectral domain, as opposed to traditional speech enhancement where all operations are often done directly on the spectrum. In this paper, we aim to investigate the role of residual and highway connections in deep neural networks for speech enhancement, and verify whether or not they operate similarly to their traditional, digital signal processing counterparts. We visualize the outputs of such connections, projected back to the spectral domain, in models trained for speech denoising, and show that while skip connections do not necessarily improve performance with regards to the number of parameters, they make speech enhancement models more interpretable. Highway BID7 and residual networks BID1 have been proposed with the objective of improving activation and gradient flow in the training of deep neural networks. On the other hand, in tasks like image reconstruction or speech enhancement, the use of such skip connections serves a different purpose: if we model a corrupted signal x = y + n as the addition of noise n to a clean signal y and x is the input to a neural network, we know that the task at hand is to predict n. In other words, to predict y, we have to alter the input x by subtracting n.In speech enhancement, the two more commonly used approaches are spectral subtraction and spectral masking. In the first, a statistical model of n is used to predict its magnitude spectrum N , which is then subtracted from the input spectrum X to yield a clean magnitude spectrum estimateŶ . In spectral masking, instead of performing subtraction, we find a multiplicative mask M which aims at either blocking time-frequency cells dominated by noise (in the case of binary masks) or scaling down energies in such time-frequency cells to make them match that of the original clean signal. Recent work in speech enhancement has explored skip connections as a way of performing masking BID4 and spectral estimation BID6 . Time domain approaches, such as SEGAN BID5 , use a UNet-style network which employs multiple skip connections as well. Other works, such as BID12 , perform spectral masking but Figure 1: Diagrams for highway, residual, and masking blocks used in this paper learn how to estimate an ideal mask instead of having the masking mechanism embedded in the neural network as a skip connection.For better understanding of such models, we would like to understand whether there are any parallels between such connections and two traditional DSP approaches to speech enhancement, namely spectral subtraction and spectral masking. We also want to understand whether models using skip connections perform better for enhancement when such connections appear only once (resembling their DSP counterparts) or repeated as multiple blocks (like in highway and residual networks). This paper shows early results of our investigation on the role of skip connections in speech enhancement models. Our preliminary experiments show that, although they have no significant impact in the performance of the models, such connections might help making the models more interpretable, as we can identify the contribution of each individual layer to the task. In the future, we intend to investigate more complex models, such as models based on the UNet architecture, as well as models that employ a temporal context window at the input instead of a single frame (such as the work in BID6 ), since those are more in line with state-of-the-art models in the literature.""",
            """Residual skip connections play important role many current generative models. Although theoretical numerical advantages are understood role speech enhancement systems has not been investigated far. When performing spectral speech enhancement residual connections are very similar nature spectral subtraction which is the one commonly employed speech enhancement approaches. Highway networks hand can be seen combination spectral masking spectral subtraction. However when using deep neural networks operations would normally happen transformed spectral domain opposed traditional speech enhancement where all operations are often done directly spectrum. paper aim investigate role residual highway connections deep neural networks speech enhancement verify whether operate similarly traditional digital signal processing counterparts. visualize outputs connections projected back spectral domain models trained speech denoising show skip connections do not necessarily improve performance regards number parameters make speech enhancement models interpretable. Highway BID7 residual networks BID1 have been proposed objective improving activation gradient flow training deep neural networks. hand tasks like image reconstruction speech enhancement use skip connections serves different purpose:if we model corrupted signalx+n addition noisen clean signal x is the input neural network know task hand is to predictn. words predict have alter inputx subtracting n.In speech enhancement two commonly used approaches are spectral subtraction spectral masking. first statistical model n is used predict magnitude spectrumN which is then subtracted input spectrumX yield clean magnitude spectrum estimateŶ. spectral masking instead performing subtraction find multiplicative mask which aims either blocking time-frequency cells dominated noise case binary masks scaling energies time-frequency cells make match original clean signal. Recent work speech enhancement has explored skip connections way performing masking BID4 spectral estimation BID6. Time domain approaches SEGAN BID5 use UNet-style network which employs multiple skip connections well. works BID12 perform spectral masking Figure1:Diagrams highway residual masking blocks used paper learn how to estimate ideal mask instead masking mechanism embedded neural network skip connection.For better understanding models would like understand whether are any parallels connections two traditional DSP approaches speech enhancement namely spectral subtraction spectral masking. also want understand whether models using skip connections perform better enhancement when such connections appear resembling DSP counterparts repeated multiple blocks like highway residual networks paper shows early results investigation role skip connections speech enhancement models. preliminary experiments show although have no significant impact performance models connections might help making models interpretable can identify contribution individual layer task. future intend investigate complex models models based UNet architecture well models employ temporal context window input instead single frame work BID6 since are more in line state-of-the-art models literature.""",
        ),
        # GSM8K
        (
            "It takes 20 minutes for the oil to heat up to 300 degrees.  It then takes 40% longer for the oil to heat up to the desired temperature of 400 degrees.  After warming the oil it takes 5 minutes less time to cook than it took to warm up the oil.  How much time passes from starting the oil to having cooked chicken?",
            "takes 20 minutes oil heat 300 degrees. takes40% longer oil heat desired temperature 400 degrees. warming oil takes 5 minutes less time cook took warm oil. How much time passes starting oil cooked chicken?",
        ),
        # boolq
        (
            "Modified-release dosage is a mechanism that (in contrast to immediate-release dosage) delivers a drug with a delay after its administration (delayed-release dosage) or for a prolonged period of time (extended-release (ER, XR, XL) dosage) or to a specific target in the body (targeted-release dosage). is modified release the same as prolonged release",
            "Modified-release dosage is a mechanism contrast immediate-release dosage delivers drug delay administration delayed-release dosage prolonged period time extended-release ER XR XL dosage specific target body targeted-release dosage is modified release prolonged release",
        ),
        # SQuAD
        (
            'Tesla claimed to have developed his own physical principle regarding matter and energy that he started working on in 1892, and in 1937, at age 81, claimed in a letter to have completed a "dynamic theory of gravity" that "[would] put an end to idle speculations and false conceptions, as that of curved space." He stated that the theory was "worked out in all details" and that he hoped to soon give it to the world. Further elucidation of his theory was never found in his writings.:309 What shape of space did Tesla consider a "false conception"?',
            "Tesla claimed have developed physical principle regarding matter energy started working 1892 1937 age81 claimed letter have completed dynamic theory gravity would put end idle speculations false conceptions curved space. stated theory was worked details hoped soon give world. elucidation theory was never found writings.:309 What shape space did Tesla consider false conception?",
        ),
    ]

    for original, expected in test_cases:
        compressed = axcer_instance.compress_prompt(original)
        print("compressed is", compressed[0])
        print("expected myf", expected)
        assert compressed[0] == expected

    assert METRICS_FILE.exists(), "test.csv was not created"

    with METRICS_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    last_rows = rows[-4:]
    assert len(last_rows) == 4

    expected_metrics = [
        {"compression_ratio": 1.587, "prompt_tokens": 746, "compressed_tokens": 470, "tokens_saved": 276},
        {"compression_ratio": 1.727, "prompt_tokens": 76, "compressed_tokens": 44, "tokens_saved": 32},
        {"compression_ratio": 1.919, "prompt_tokens": 71, "compressed_tokens": 37, "tokens_saved": 34},
        {"compression_ratio": 1.833, "prompt_tokens": 121, "compressed_tokens": 66, "tokens_saved": 55},
    ]

    for row, expected in zip(last_rows, expected_metrics):
        assert float(row["compression_ratio"]) == expected["compression_ratio"]
        assert int(row["prompt_tokens"]) == expected["prompt_tokens"]
        assert int(row["compressed_tokens"]) == expected["compressed_tokens"]
