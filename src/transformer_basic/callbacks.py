from lightning.pytorch.callbacks import RichProgressBar

class CustomRichProgressBar(RichProgressBar):
    """
    A custom RichProgressBar that handles a known IndexError on teardown.
    This issue can occur in certain environments when the progress bar's
    live display is stopped more often than it was started.
    """
    def teardown(self, trainer, pl_module, stage: str) -> None:
        try:
            super().teardown(trainer, pl_module, stage)
        except IndexError:
            # This is a known issue in some environments where the progress
            # bar's live display is stopped more than once. We can safely
            # ignore it as the training is already complete.
            pass
