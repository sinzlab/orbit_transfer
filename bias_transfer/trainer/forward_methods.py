import torch
from torch.nn import functional as F

from bias_transfer.trainer.main_loop_modules import RDL


def ce_forward(model, x, y, *args, **kwargs):
    return F.cross_entropy(model(x)[1], y, reduction="mean"), torch.tensor(0)


def equiv_transfer_forward(model, x, y, teacher_model, config=None):
    b = x.shape[0]
    x = x.reshape(b, 1, -1)
    ### encourage equivariance ###
    # sample group element (one per batch element)
    g = torch.randint(40, (x.shape[0],))
    # n = torch.randint(1, 3, (1,))
    n = 1
    # apply group representation on input
    rho_g_x = teacher_model(x, g)
    # pass transformed and non-transformed input through the model
    x_combined = torch.cat([x, rho_g_x], dim=0)
    layer_out, final_out = model(x_combined)
    layers = list(layer_out.keys())
    # collect output that for transformed input
    phi_rho_g_x = torch.cat(
        [layer_out[l][b:].flatten(1) for l in layers] + [final_out[b:].flatten(1)],
        dim=1,
    )
    # apply group representation on output of untransformed input
    rho_g_phi_x = torch.cat(
        [
            teacher_model(layer_out[layer][:b], g, l, n).flatten(1)
            for l, layer in enumerate(layers)
        ]
        + [teacher_model(final_out[:b], g, len(layers), n).flatten(1)],
        dim=1,
    )
    # minimize distance
    regularizer = F.mse_loss(rho_g_phi_x, phi_rho_g_x)

    # also get CE loss for transformed input
    regularizer += F.cross_entropy(final_out.flatten(1)[b:], y, reduction="mean")

    main_loss = F.cross_entropy(final_out.flatten(1)[:b], y, reduction="mean")

    return main_loss, regularizer


def equiv_learn_forward(model, x, y, teacher_model, config):
    b = x.shape[0]
    x = x.reshape(b, 1, -1)
    ### encourage equivariance ###
    # sample group element (one per batch element)
    g = torch.randint(0, 40, (x.shape[0],))
    # n = torch.randint(1, 3, (1,))
    n = 1
    h = (g + torch.randint(1, 40, (x.shape[0],))) % 40
    # apply group representation on input
    transformed_x = model(x.repeat(2, 1, 1), torch.cat([g, h], dim=0))
    rho_g_x, rho_h_x = (
        transformed_x[:b],
        transformed_x[b:],
    )
    # pass transformed and non-transformed input through the model
    x_combined = torch.cat([x, rho_g_x], dim=0)
    out_combined, final_out = teacher_model(x_combined)
    layers = list(out_combined.keys())
    # collect output that for transformed input
    phi_rho_g_x = torch.cat(
        [out_combined[l][b:].flatten(1) for l in layers] + [final_out[b:].flatten(1)],
        dim=1,
    )
    # apply group representation on output of untransformed input
    rho_g_phi_x = torch.cat(
        [
            model(out_combined[layer][:b], g, l, n).flatten(1)
            for l, layer in enumerate(layers)
        ]
        + [model(final_out[:b], g, len(layers), n).flatten(1)],
        dim=1,
    )
    # minimize distance
    loss = F.mse_loss(rho_g_phi_x, phi_rho_g_x) * config.equiv_factor

    ### enforce invertible ###
    inv_rho_g_x = model(rho_g_x, -g, n=n)
    loss += F.mse_loss(x.squeeze(), inv_rho_g_x.squeeze()) * config.invertible_factor

    ### prevent identity solution ###
    loss += (
        torch.abs(
            F.cosine_similarity(rho_h_x.flatten(1), rho_g_x.flatten(1), dim=1, eps=1e-8)
        ).mean()
        * config.identity_factor
    )  # minimize similarity by adding to the loss
    return loss, torch.tensor(0.0)


def kd_forward(model, x, y, teacher_model, config):
    model_out = model(x)
    return (
        F.cross_entropy(model_out[1], y, reduction="mean"),
        F.kl_div(
            F.log_softmax(model_out[1] / config.softmax_temp, dim=1),
            F.softmax(teacher_model(x)[1] / config.softmax_temp, dim=1),
            reduction="batchmean",  # batchmean?
        )
        * config.softmax_temp ** 2,
    )


def kd_match_forward(model, x, y, teacher_model, config):
    model_out = model(x)
    teacher_out = teacher_model(x)
    match = 0.0
    for layer in model_out[0].keys():
        teacher_layer = layer.replace("linear", "conv")
        match += F.mse_loss(
            model_out[0][layer].flatten(1),
            teacher_out[0][teacher_layer].flatten(1),
            reduction="mean",
        )
    match += F.kl_div(
        F.log_softmax(model_out[1] / config.softmax_temp, dim=1),
        F.softmax(teacher_out[1] / config.softmax_temp, dim=1),
        reduction="mean",
    )
    return (
        F.cross_entropy(model_out[1], y, reduction="mean"),
        match,
    )


def compute_rdm(x, dist_measure="corr"):
    x_flat = x.flatten(1, -1)
    centered = x_flat - x_flat.mean(dim=0).view(1, -1)  # centered by mean over images
    result = (centered @ centered.transpose(0, 1)) / torch.ger(
        torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
    )  # see https://de.mathworks.com/help/images/ref/corr2.html
    return result


def rdl_forward(model, x, y, teacher_model, config):
    model_out = model(x)
    teacher_out = teacher_model(x)
    match = 0.0
    for layer in model_out[0].keys():
        teacher_layer = layer.replace("linear", "conv")
        rdm_s = compute_rdm(model_out[0][layer])
        rdm_t = compute_rdm(teacher_out[0][teacher_layer])
        match += F.mse_loss(rdm_s.flatten(), rdm_t.flatten(), reduction="mean")
    rdm_s = compute_rdm(model_out[1])
    rdm_t = compute_rdm(teacher_out[1])
    match += F.mse_loss(rdm_s.flatten(), rdm_t.flatten(), reduction="mean")
    return F.cross_entropy(model_out[1], y, reduction="mean"), match


def cka_forward(model, x, y, teacher_model, config):
    model_out = model(x)
    teacher_out = teacher_model(x)
    match = 0.0
    for layer in model_out[0].keys():
        teacher_layer = layer.replace("linear", "conv")
        match -= RDL.linear_CKA(
            model_out[0][layer].flatten(1), teacher_out[0][teacher_layer].flatten(1)
        )
    match -= RDL.linear_CKA(model_out[1], teacher_out[1])
    return F.cross_entropy(model_out[1], y, reduction="mean"), match


def attention_forward(model, x, y, teacher_model, config):
    model_out = model(x)
    teacher_out = teacher_model(x)
    match = 0.0
    for layer in model_out[0].keys():
        teacher_layer = layer.replace("linear", "conv")
        attention_map_t = torch.sum(
            torch.abs(teacher_out[0][teacher_layer]) ** 2, dim=1
        ).flatten(1)
        b, s = attention_map_t.shape
        attention_map_s = torch.sum(
            (torch.abs(model_out[0][layer]) ** 2).reshape(b, -1, s), dim=1
        ).flatten(1)
        attention_map_t = attention_map_t / torch.norm(
            attention_map_t, dim=1, keepdim=True
        )
        attention_map_s = attention_map_s / torch.norm(
            attention_map_s, dim=1, keepdim=True
        )
        match += F.mse_loss(attention_map_s, attention_map_t, reduction="mean")
    return F.cross_entropy(model_out[1], y, reduction="mean"), match
