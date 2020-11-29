import torch
import torch.nn as nn
import torchvision.models as models
import heapq

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, params.embed_size)        # Do we need to dropout between resnet and dropout?

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class Decoder(nn.Module):
    def __init__(self, params, vocab_size):
        super(Decoder, self).__init__()
        self.params = params
        # batch_first-If True, then the input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=params.embed_size, hidden_size=params.hidden_size, num_layers=params.LSTM_layers,
                            dropout=params.LSTM_dropout, batch_first=True)
        self.embed = nn.Embedding(vocab_size, params.embed_size)
        self.fc = nn.Linear(params.hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=0)


    def forward(self, features, captions):
        features = features.unsqueeze(1)
        captions = captions[:, :-1]
        captions_emb = self.embed(captions)
        input_ = torch.cat((features, captions_emb), 1)
        out, _ = self.lstm(input_)      # don't need the 2nd parameter (h, c) here
        out = self.fc(out)
        return out


    def predict(self, inputs):
        """ accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len)"""
        hc = (torch.zeros(size=[self.params.LSTM_layers, 1, self.params.hidden_size], device=self.params.device),
              torch.zeros(size=[self.params.LSTM_layers, 1, self.params.hidden_size], device=self.params.device))

        # according to "Show and Tell: A Neural Image Caption Generator", BeamSearch works better
        if self.params.inference_mode == 'sampling':
            tokens = []
            for _ in range(self.params.maxlen_generate):
                out, hc = self.lstm(inputs, hc)
                out = self.fc(out)
                out = out.squeeze(1)
                pred = out.argmax(1)        # format: tensor([0], device='cuda:0')
                tokens.append(pred.item())
                inputs = self.embed(pred)
                inputs = inputs.unsqueeze(0)
            return tokens

        elif self.params.inference_mode == 'beamsearch':
            num_candidates = self.params.num_candidates
            # the candidates for beamsearch. at most num_candidates of tuple, each of which contains
            # (probability for current sequence, word indices sequence, embedding for last word, output hc of last word)
            candidates = [(1, [], inputs, hc)]
            for _ in range(self.params.maxlen_generate):
                scores_id = []
                for former_prob, ind_sequence, emb, hc in candidates:
                    out, hc = self.lstm(emb, hc)
                    out = self.fc(out)
                    out = out.squeeze()
                    out = self.softmax(out)

                    # in extreme case all candidates come from the same former word
                    score_heap = []
                    for i, score in enumerate(out):
                        if len(score_heap) < num_candidates:    # when heap is not full, push
                            heapq.heappush(score_heap, (score, i))
                        elif score > score_heap[0][0]:          # if it's full, only push when it has a higher score than top of min-heap
                            heapq.heappushpop(score_heap, (score, i))

                    for score, i in score_heap:
                        if len(scores_id) < num_candidates:
                            heapq.heappush(scores_id, (former_prob * score, ind_sequence.copy() + [i], i, hc))
                        elif former_prob * score > scores_id[0][0]:
                            heapq.heapreplace(scores_id, (former_prob * score, ind_sequence.copy() + [i], i, hc))

                # select candidates for next iteration, and calculate embed
                candidates = []
                max_prob = -float('inf')
                for prob, seq, i, hc in scores_id:
                    max_prob = max(max_prob, prob)     # avoid probability multi-product to zero
                    # self.embed requires a tensor as input rather than int
                    candidates.append((prob, seq, self.embed(torch.Tensor([i]).long().to(self.params.device)).unsqueeze(0), hc))

                for j in range(len(candidates)):
                    candidates[j] = (candidates[j][0]/max_prob, candidates[j][1], candidates[j][2], candidates[j][3])

            # find the sequence with highest score (probability)
            max_score, tokens = -float('inf'), []
            for prob, seq, _, _ in candidates:
                if prob > max_score:
                    max_score = prob
                    tokens = seq
            return tokens

        else:
            print("Invalid inference_mode in params class. Must be 'sampling' or 'beamsearch'.")
