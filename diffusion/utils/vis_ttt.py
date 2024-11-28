"""
Visualize TTT inner losses into wandb
"""
import wandb
import jax.numpy as jnp
import matplotlib.pyplot as plt


def vis_ttt_lm(aux, log_n_plots=4):
    """ Dig out the TTT inner losses and visualize them into wandb. """
    if "aux" not in aux or "dit_blocks" not in aux["aux"]:
        return
    dit_blocks = aux["aux"]["dit_blocks"]
    for i, dit_block in enumerate(dit_blocks):
        if "attn.forward" in dit_block and "output_stats" in dit_block["attn.forward"]:
            attn_forward = dit_block["attn.forward"]["output_stats"]
            ttt_loss_mse_init = attn_forward["ttt_loss_mse_init"]
            ttt_loss_mse_step0 = attn_forward["ttt_loss_mse_step0"]
            ttt_loss_mse_step1 = attn_forward["ttt_loss_mse_step1"]
            
            fig = plt.figure()
            plt.plot(
                range(len(ttt_loss_mse_init)), ttt_loss_mse_init, label="init")
            plt.plot(
                range(len(ttt_loss_mse_step0)), ttt_loss_mse_step0, label="step0")
            plt.plot(
                range(len(ttt_loss_mse_step1)), ttt_loss_mse_step1, label="step1")
            wandb.log(
                {f"inner_loss/plot_forward_layer{i}": fig}, 
                commit=False)
            for j in [0, len(ttt_loss_mse_init)//2, len(ttt_loss_mse_init) - 1]:
                wandb.log({
                    f"inner_loss/layer{i}_step{j}_forward_mse_init": ttt_loss_mse_init[j],
                    f"inner_loss/layer{i}_step{j}_forward_mse_step0": ttt_loss_mse_step0[j],
                    f"inner_loss/layer{i}_step{j}_forward_mse_step1": ttt_loss_mse_step1[j],
                }, commit=False)
                
        if "attn.forward" in dit_block and "input_stats" in dit_block["attn.forward"]:
            attn_forward = dit_block["attn.forward"]["input_stats"]
            eta_avg = attn_forward["eta"].mean()
            wandb.log({
                f"inner_loss/layer{i}_step{j}_forward_eta_avg": eta_avg,
            }, commit=False)
                
        if "attn.backward" in dit_block and "output_stats" in dit_block["attn.backward"]:
            attn_backward = dit_block["attn.backward"]["output_stats"]
            ttt_loss_mse_init = attn_backward["ttt_loss_mse_init"]
            ttt_loss_mse_step0 = attn_backward["ttt_loss_mse_step0"]
            ttt_loss_mse_step1 = attn_backward["ttt_loss_mse_step1"]
            
            fig = plt.figure()
            plt.plot(
                range(len(ttt_loss_mse_init)), ttt_loss_mse_init, label="init")
            plt.plot(
                range(len(ttt_loss_mse_step0)), ttt_loss_mse_step0, label="step0")
            plt.plot(
                range(len(ttt_loss_mse_step1)), ttt_loss_mse_step1, label="step1")
            wandb.log(
                {f"inner_loss/plot_backward_layer{i}": fig}, 
                commit=False)
            for j in [0, len(ttt_loss_mse_init)//2, len(ttt_loss_mse_init) - 1]:
                wandb.log({
                    f"inner_loss/layer{i}_step{j}_backward_mse_init": ttt_loss_mse_init[j],
                    f"inner_loss/layer{i}_step{j}_backward_mse_step0": ttt_loss_mse_step0[j],
                    f"inner_loss/layer{i}_step{j}_backward_mse_step1": ttt_loss_mse_step1[j],
                }, commit=False)
    
        if "attn.backward" in dit_block and "input_stats" in dit_block["attn.backward"]:
            attn_backward = dit_block["attn.backward"]["input_stats"]
            eta_avg = attn_backward["eta"].mean()
            wandb.log({
                f"inner_loss/layer{i}_step{j}_backward_eta_avg": eta_avg,
            }, commit=False)
   
    
def vis_ttt_orig(aux):
    """ Dig out the TTT inner losses and visualize them into wandb. 
        For the simplified version of the TTT model.                """
    pass
    # if "aux" not in aux or "dit_block" not in aux: return
    # for i, dit_block in enumerate(aux["aux"]["dit_bock"]):
    # if ("aux" in aux and len(aux["aux"]) > 0 
    #     and "attn.inner_losses" in aux["aux"][0]):
    #     aux_dict = aux["aux"]
    #     inner_losses_bf = [
    #         jnp.array(aux_dict[str(i)]["attn.inner_losses"]["bf"]).mean(
    #             axis=list(range(
    #                 len(jnp.array(aux_dict[str(i)]["attn.inner_losses"]["bf"]).shape[:-1])
    #             )))
    #         for i in range(len(aux["aux"]))]
    #     inner_losses_af = [
    #         jnp.array(aux["aux"][i]["attn.inner_losses"]["af"]).mean(
    #             axis=list(range(
    #                 len(jnp.array(aux["aux"][i]["attn.inner_losses"]["bf"]).shape[:-1])
    #             )))
    #         for i in range(len(aux["aux"]))]
    #     n_layers = len(aux["aux"])
    #     for i in range(n_layers):
    #         fig = plt.figure()
    #         plt.plot(
    #             range(len(inner_losses_bf[i])),
    #             inner_losses_bf[i], label="bf"
    #         )
    #         plt.plot(
    #             range(len(inner_losses_af[i])),
    #             inner_losses_af[i], label="af"
    #         )
    #         wandb.log(
    #             {f"inner_loss/plot_layer{i}": fig}, 
    #             commit=False)
    #         wandb.log({
    #             f"inner_loss/loss_bf_layer{i}_step{j}": lv 
    #             for j, lv in enumerate(inner_losses_bf[i])
    #         }, commit=False)
    #         wandb.log({
    #             f"inner_loss/loss_af_layer{i}_step{j}": lv 
    #             for j, lv in enumerate(inner_losses_af[i])
    #         }, commit=False)