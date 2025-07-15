import torch
import torch.nn as nn
import torch.nn.functional as F  # For functions like pad, interpolate, softmax

from torchvision.models import resnet18
from modules.encoders import get_encoder, MMILB, CPC, SubNet
from resnet import MyResNet, BasicBlock


class FeatureAttention(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        scores = self.attention_scorer(x)
        attn_weights = self.softmax(scores)
        attended_output = x * attn_weights
        return attended_output


class MMIM(nn.Module):
    def __init__(self, hp):
        super().__init__()
        print('hp:', hp)
        self.hp = hp
        self.add_va = hp.add_va
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_visual = getattr(hp, 'use_visual', True)
        self.use_audio = getattr(hp, 'use_audio', True)

        self.frame_encoder = MyResNet(BasicBlock, [2, 2, 2, 2]).eval()
        self.load_vision_weight()
        self.frame_encoder = self.frame_encoder.to(self.device)

        if self.use_visual:
            self.visual_enc = get_encoder(
                encoder_type=hp.visual_encoder_type,
                in_size=hp.d_vin + hp.d_pose,
                hidden_size=hp.d_vh,
                out_size=hp.d_vout,
                num_layers=hp.n_layer,
                dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
                bidirectional=hp.bidirectional
            ).to(self.device)

        if self.use_audio:
            self.audio_enc = get_encoder(
                encoder_type=hp.acoustic_encoder_type,
                in_size=hp.d_audio_in + hp.d_is09_in,
                hidden_size=hp.d_ah,
                out_size=hp.d_aout,
                num_layers=hp.n_layer,
                dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
                bidirectional=hp.bidirectional
            ).to(self.device)

        self.mi_va = MMILB(
            x_size=hp.d_vout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        ).to(self.device)

        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        ).to(self.device)

        self.use_transformer_fusion = getattr(hp, 'use_transformer_fusion', False)
        self.use_attention_fusion = getattr(hp, 'use_attention_fusion', False)

        if self.use_transformer_fusion and self.use_attention_fusion:
            raise ValueError("Cannot use both Transformer and Attention fusion simultaneously.")

        fusion_input_size = 0
        if self.use_visual:
            fusion_input_size += hp.d_vout
        if self.use_audio:
            fusion_input_size += hp.d_aout

        self.transformer_fusion = None
        self.attention_fusion = None

        if self.use_transformer_fusion:
            print("Using Transformer Fusion")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_input_size,
                nhead=getattr(hp, 'transformer_nhead', 4),
                dim_feedforward=hp.d_prjh,
                dropout=hp.dropout_prj,
                activation='relu',
                batch_first=True
            )
            transformer_layers = getattr(hp, 'transformer_fusion_layers', 1)
            self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers).to(self.device)

        elif self.use_attention_fusion:
            print("Using Attention Fusion")
            self.attention_fusion = FeatureAttention(
                d_model=fusion_input_size,
                hidden_dim=hp.d_prjh
            ).to(self.device)

        self.fusion_prj = SubNet(
            in_size=fusion_input_size,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        ).to(self.device)

    def load_vision_weight(self):
        try:
            resnet = resnet18(weights='IMAGENET1K_V1')
        except:
            try:
                resnet = resnet18(pretrained=True)
            except:
                try:
                    resnet = resnet18(pretrained=False)
                except:
                    resnet = resnet18(weights=None)

        pretrained_dict = resnet.state_dict()
        model_dict = self.frame_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.frame_encoder.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} matching keys into frame encoder.")

        for _, p in self.frame_encoder.named_parameters():
            p.requires_grad = False

    def forward(self, visual, pose_features, audio_features, is09_features, y=None):
        device = self.device
        batch_size = visual.size(0)
        seq_len = visual.size(1)

        fusion_inputs = []

        if self.use_visual:
            visual = visual.to(device)
            pose_features = pose_features.to(device)
            visual_features = [self.frame_encoder(visual[:, t]).view(batch_size, -1) for t in range(seq_len)]
            visual_features = torch.stack(visual_features, dim=1)
            pose_features = pose_features.view(batch_size, seq_len, -1)
            pose_features = self._pad_or_truncate(pose_features, self.hp.d_pose)
            combined_visual = torch.cat((visual_features, pose_features), dim=-1)
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)
            visual_encoded_output = self.visual_enc(combined_visual, lengths)
            visual_pooled = visual_encoded_output.mean(dim=1) if visual_encoded_output.ndim == 3 else visual_encoded_output
            fusion_inputs.append(visual_pooled)

        if self.use_audio:
            audio_features = audio_features.to(device)
            is09_features = is09_features.to(device)
            if audio_features.dim() == 4:
                audio_features = audio_features.squeeze(-1) if audio_features.shape[-1] == 1 else audio_features.squeeze(-2)
            current_seq_len = audio_features.size(1)

            if is09_features.dim() == 2:
                is09_features = is09_features.unsqueeze(1).expand(-1, current_seq_len, -1)
            elif is09_features.size(1) != current_seq_len:
                is09_features = F.interpolate(is09_features.permute(0, 2, 1), size=current_seq_len, mode='nearest').permute(0, 2, 1)

            combined_audio = torch.cat((audio_features, is09_features), dim=-1)
            combined_audio = self._pad_or_truncate(combined_audio, self.hp.d_audio_in + self.hp.d_is09_in)
            audio_lengths = torch.full((batch_size,), current_seq_len, dtype=torch.long).to(device)
            audio_encoded_output = self.audio_enc(combined_audio, audio_lengths)
            audio_pooled = audio_encoded_output.mean(dim=1) if audio_encoded_output.ndim == 3 else audio_encoded_output
            fusion_inputs.append(audio_pooled)

        fusion_input = torch.cat(fusion_inputs, dim=1)
        fused_features = fusion_input

        if self.transformer_fusion is not None:
            fused_features = self.transformer_fusion(fusion_input.unsqueeze(1)).squeeze(1)
        elif self.attention_fusion is not None:
            fused_features = self.attention_fusion(fusion_input)

        output = self.fusion_prj(fused_features)
        return output

    def _pad_or_truncate(self, features, target_dim):
        current_dim = features.size(-1)
        if current_dim < target_dim:
            features = F.pad(features, (0, target_dim - current_dim))
        elif current_dim > target_dim:
            features = features[..., :target_dim]
        return features
