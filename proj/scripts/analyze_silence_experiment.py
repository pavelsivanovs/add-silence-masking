import argparse
import logging

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

logger = logging.getLogger(__name__)

OLD_FORMULA = "correct ~ duration_s + position"
FORMULA = "correct ~ duration_s * true_label + position * true_label + utterance_length_s"


def fit_model(path: str, lengths_path: str, label: str) -> dict:
    df = pd.read_csv(path, sep="\t")
    lengths = pd.read_csv(lengths_path, sep="\t")
    df = df.merge(lengths, on="audio_file_name", how="left")
    if df["utterance_length_s"].isna().any():
        missing = df.loc[df["utterance_length_s"].isna(), "audio_file_name"].unique()
        raise ValueError(f"[{label}] missing utterance_length_s for {len(missing)} file(s)")

    gee = smf.gee(
        FORMULA,
        groups="audio_file_name",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()

    old_logit = smf.logit(OLD_FORMULA, data=df).fit(disp=False)
    new_logit = smf.logit(FORMULA, data=df).fit(disp=False)

    try:
        bayes = BinomialBayesMixedGLM.from_formula(
            FORMULA, {"speaker": "0 + C(speaker_id)"}, data=df
        ).fit_vb()
        bayes_summary = bayes.summary()
    except Exception as e:
        logger.warning(f"[{label}] BinomialBayesMixedGLM cross-check failed to converge: {e}")
        bayes_summary = None

    print(f"===== {label} =====")
    print(f"N = {len(df)}, speakers = {df.speaker_id.nunique()}, utterances = {df.audio_file_name.nunique()}")
    print(f"Model comparison (unclustered logit): old vs. new formula")
    print(f"old ({OLD_FORMULA}): AIC={old_logit.aic:.1f}, pseudo-R^2={old_logit.prsquared:.4f}")
    print(f"new ({FORMULA}): AIC={new_logit.aic:.1f}, pseudo-R^2={new_logit.prsquared:.4f}")
    print("GEE (Binomial, exchangeable, clustered by audio_file_name)")
    print(gee.summary())
    if bayes_summary is not None:
        print("BinomialBayesMixedGLM (speaker random intercept) cross-check")
        print(bayes_summary)

    return {
        "label": label,
        "n": len(df),
        "n_speakers": df.speaker_id.nunique(),
        "n_utterances": df.audio_file_name.nunique(),
        "old_aic": old_logit.aic,
        "new_aic": new_logit.aic,
        "old_mcfadden_r2": old_logit.prsquared,
        "new_mcfadden_r2": new_logit.prsquared,
        "gee_params": gee.params.to_dict(),
        "gee_bse": gee.bse.to_dict(),
        "gee_pvalues": gee.pvalues.to_dict(),
    }


def main(inputs: list[tuple[str, str]], lengths_path: str) -> list[dict]:
    return [fit_model(path, lengths_path, label) for label, path in inputs]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(prog="Fit the silence-masking regression.")
    parser.add_argument("--mean-tsv", default="artifacts/silence_experiment_default_mean.tsv")
    parser.add_argument("--max-tsv", default="artifacts/silence_experiment_default_max.tsv")
    parser.add_argument("--lengths-tsv", default="artifacts/utterance_lengths.tsv")
    args = parser.parse_args()

    main(
        [("Mean pooling", args.mean_tsv), ("Max pooling", args.max_tsv)],
        args.lengths_tsv,
    )
