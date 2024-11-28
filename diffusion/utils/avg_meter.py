
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def add_to_sum(self, sum_val):
        self.sum += float(sum_val)
        self.avg = float(self.sum / float(self.count))
    
    def add_to_count(self, count_val):
        self.count += float(count_val)
        self.avg = float(self.sum / float(self.count))

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val * n)
        self.count += float(n)
        self.avg = float(self.sum / float(self.count))

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)