
function run_linf() {
    # cifar10
    python3 -u main.py --model PreActResNet18 --config cifar10_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config cifar10_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config cifar10_linf_AWP_AT --project at --gpu 2 &&

    python3 -u main.py --model WideResNet28 --config cifar10_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config cifar10_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config cifar10_linf_AWP_AT --project at --gpu 2 &&

    # cifar100
    python3 -u main.py --model PreActResNet18 --config cifar100_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config cifar100_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config cifar100_linf_AWP_AT --project at --gpu 2 &&

    python3 -u main.py --model WideResNet28 --config cifar100_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config cifar100_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config cifar100_linf_AWP_AT --project at --gpu 2 &&

    # SVHN
    python3 -u main.py --model PreActResNet18 --config svhn_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config svhn_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model PreActResNet18 --config svhn_linf_AWP_AT --project at --gpu 2 &&

    python3 -u main.py --model WideResNet28 --config svhn_linf_AT --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config svhn_linf_TRADES --project at --gpu 2 &&
    python3 -u main.py --model WideResNet28 --config svhn_linf_AWP_AT --project at --gpu 2 &&

    echo "Done"
}

function run_l2() {
    # cifar10
    python3 -u main.py --model PreActResNet18 --config cifar10_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config cifar10_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config cifar10_l2_AWP_AT --project at --gpu 3 &&

    python3 -u main.py --model WideResNet28 --config cifar10_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config cifar10_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config cifar10_l2_AWP_AT --project at --gpu 3 &&

    # cifar100
    python3 -u main.py --model PreActResNet18 --config cifar100_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config cifar100_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config cifar100_l2_AWP_AT --project at --gpu 3 &&

    python3 -u main.py --model WideResNet28 --config cifar100_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config cifar100_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config cifar100_l2_AWP_AT --project at --gpu 3 &&

    # SVHN
    python3 -u main.py --model PreActResNet18 --config svhn_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config svhn_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model PreActResNet18 --config svhn_l2_AWP_AT --project at --gpu 3 &&

    python3 -u main.py --model WideResNet28 --config svhn_l2_AT --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config svhn_l2_TRADES --project at --gpu 3 &&
    python3 -u main.py --model WideResNet28 --config svhn_l2_AWP_AT --project at --gpu 3 &&

    echo "Done"
}

if [ "$1"x == "l2"x ]
then
    echo "L_2"
    run_l2
else
    echo "L_inf"
    run_linf
fi
