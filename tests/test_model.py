import unittest

import numpy as np

from cubecast import CubeCast


def tensor(start=0, length=72):
    time = np.arange(start, start + length)
    season = np.sin(2 * np.pi * time / 12)
    values = np.empty((length, 4, 2))
    for location in range(4):
        group = -1 if location < 2 else 1
        values[:, location, 0] = 0.02 * time + group * season
        values[:, location, 1] = -0.01 * time + 0.5 * group * season
    return values


class CubeCastTest(unittest.TestCase):
    def test_fit_and_predict(self):
        model = CubeCast(period=12, window_size=48, max_components=2)
        forecast = model.fit_predict(tensor(), 6)

        self.assertEqual(forecast.shape, (6, 4, 2))
        self.assertTrue(np.all(np.isfinite(forecast)))
        self.assertGreaterEqual(model.n_regimes_, 1)
        self.assertEqual(sum(map(len, model.groups_)), 4)
        self.assertEqual(len(model.groups_), 2)
        self.assertEqual(type(model._season_ica).__name__, "FastICA")
        self.assertEqual(model.transition_optimizer_, "lmfit.leastsq")

    def test_update_keeps_public_state_consistent(self):
        model = CubeCast(period=12, window_size=48, max_components=2)
        model.fit(tensor(length=60)).update(tensor(start=60, length=12))

        self.assertEqual(model.predict(3).shape, (3, 4, 2))
        self.assertLess(model.active_regime_, model.n_regimes_)

    def test_nonseasonal_model(self):
        model = CubeCast(max_components=2).fit(tensor(length=24))

        self.assertEqual(model.seasonal_components_, 0)
        self.assertIsNone(model._season_ica)
        self.assertTrue(np.all(np.isfinite(model.predict(2))))

    def test_decomposition_is_additive(self):
        model = CubeCast(period=12, window_size=48, max_components=2)
        values = model.fit(tensor()).decompose(location=1, feature=0)

        expected = values["baseline"] + values["trend"] + values["seasonal"]
        np.testing.assert_allclose(values["reconstruction"], expected)
        self.assertEqual(values["trend_latent"].shape[0], 48)
        self.assertEqual(values["seasonal_latent"].shape[0], 48)

    def test_detects_a_large_regime_change(self):
        model = CubeCast(period=12, window_size=48, max_components=2)
        model.fit(tensor(length=48))
        model.update(8 * tensor(start=48, length=12) + 20)

        self.assertEqual(model.n_regimes_, 2)
        self.assertEqual(model.active_regime_, 1)

    def test_fastica_handles_a_low_rank_seasonal_profile(self):
        time = np.arange(48)
        values = np.empty((48, 2, 2))
        values[:, :, 0] = np.sin(2 * np.pi * time[:, None] / 12) * [[1, -1]]
        values[:, :, 1] = np.cos(2 * np.pi * time[:, None] / 12) * [[0.5, 1.5]]

        model = CubeCast(period=12, max_components=2).fit(values)

        self.assertEqual(model.seasonal_components_, 1)
        self.assertTrue(np.all(np.isfinite(model.predict(4))))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            CubeCast().fit(np.ones((10, 2)))

    def test_window_must_hold_two_periods(self):
        with self.assertRaises(ValueError):
            CubeCast(period=12, window_size=20).fit(tensor())


if __name__ == "__main__":
    unittest.main()
