from .. import imf

def test_mmax():
    """
    Regression test for issue #4
    """

    c = imf.make_cluster(100000, mmax=1, mmin=0.01)

    assert c.max() <= 1
