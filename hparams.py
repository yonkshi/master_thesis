import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # train
    ## files
    parser.add_argument('--train1', default='iwslt2016/segmented/train.de.bpe',
                             help="german training segmented data")
    parser.add_argument('--train2', default='iwslt2016/segmented/train.en.bpe',
                             help="english training segmented data")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)

    # model
    parser.add_argument('--d_states', default=32, type=int,
                        help="hidden dimension of state attention module")
    parser.add_argument('--d_states_ff', default=128, type=int,
                        help="hidden dimension of feedforward layer")

    parser.add_argument('--d_actions', default=32, type=int,
                        help="hidden dimension of action attention module")
    parser.add_argument('--d_actions_ff', default=128, type=int,
                        help="hidden dimension of feedforward layer")


    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of attention blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    # test
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")