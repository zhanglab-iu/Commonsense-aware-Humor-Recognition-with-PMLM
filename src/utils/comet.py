# Importing comet-commonsense modules
import os
import sys

# sys.path.insert(0, os.getcwd() + "src/comet_commonsense")
import src.comet_commonsense.src.data.conceptnet as cdata
import src.comet_commonsense.src.data.data as data
import src.comet_commonsense.src.models.models as models
import src.comet_commonsense.utils as tmp_utils
import src.comet_commonsense.utils.utils as utils
from src.comet_commonsense.src.data.utils import TextEncoder


def load_all(model_path: str):
    sys.modules["utils"] = tmp_utils
    sys.modules["src.data.conceptnet"] = cdata
    model_stuff = data.load_checkpoint(model_path)
    opt = model_stuff["opt"]

    relations = data.conceptnet_data.conceptnet_relations

    if opt.data.get("maxr", None) is None:
        if opt.data.rel == "language":
            opt.data.maxr = 5
        else:
            opt.data.maxr = 1

    path = (
        "src/comet_commonsense/data/conceptnet/processed/generation/{}.pickle".format(
            utils.make_name_string(opt.data)
        )
    )
    data_loader = data.make_data_loader(opt)
    _ = data_loader.load_data(path)

    encoder_path = "src/comet_commonsense/model/encoder_bpe_40000.json"
    bpe_path = "src/comet_commonsense/model/vocab_40000.bpe"

    text_encoder = TextEncoder(encoder_path, bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in relations]

    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    _ = len(special)
    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = models.make_model(
        opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False
    )

    models.load_state_dict(model, model_stuff["state_dict"])
    # encoder_model = model.transformer

    return model, text_encoder, opt, data_loader


def create_comet(model_path: str, cwd: str = "."):
    sys.modules["utils"] = tmp_utils
    sys.modules["src.data.conceptnet"] = cdata
    model_stuff = data.load_checkpoint(model_path)
    opt = model_stuff["opt"]

    relations = data.conceptnet_data.conceptnet_relations

    if opt.data.get("maxr", None) is None:
        if opt.data.rel == "language":
            opt.data.maxr = 5
        else:
            opt.data.maxr = 1

    path = (
        "src/comet_commonsense/data/conceptnet/processed/generation/{}.pickle".format(
            utils.make_name_string(opt.data)
        )
    )
    path = os.path.join(cwd, path)
    data_loader = data.make_data_loader(opt)
    _ = data_loader.load_data(path)

    encoder_path = "src/comet_commonsense/model/encoder_bpe_40000.json"
    bpe_path = "src/comet_commonsense/model/vocab_40000.bpe"
    encoder_path = os.path.join(cwd, encoder_path)
    bpe_path = os.path.join(cwd, bpe_path)

    text_encoder = TextEncoder(encoder_path, bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in relations]

    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    _ = len(special)
    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = models.make_model(
        opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False
    )

    models.load_state_dict(model, model_stuff["state_dict"])
    # encoder_model = model.transformer

    return model, text_encoder


def create_text_encoder(model_path: str, cwd: str = "."):
    sys.modules["utils"] = tmp_utils
    sys.modules["src.data.conceptnet"] = cdata
    model_stuff = data.load_checkpoint(model_path)
    opt = model_stuff["opt"]

    relations = data.conceptnet_data.conceptnet_relations

    if opt.data.get("maxr", None) is None:
        if opt.data.rel == "language":
            opt.data.maxr = 5  # ここが走る
        else:
            opt.data.maxr = 1

    path = "comet_commonsense/data/conceptnet/processed/generation/{}.pickle".format(
        utils.make_name_string(opt.data)
    )
    path = os.path.join(cwd, path)
    data_loader = data.make_data_loader(opt)
    _ = data_loader.load_data(path)

    encoder_path = "comet_commonsense/model/encoder_bpe_40000.json"
    bpe_path = "comet_commonsense/model/vocab_40000.bpe"
    encoder_path = os.path.join(cwd, encoder_path)
    bpe_path = os.path.join(cwd, bpe_path)

    text_encoder = TextEncoder(encoder_path, bpe_path)

    special = [data.start_token, data.end_token]
    special += ["<{}>".format(cat) for cat in relations]

    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    _ = len(special)
    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    _ = len(text_encoder.encoder) + n_ctx

    return text_encoder
