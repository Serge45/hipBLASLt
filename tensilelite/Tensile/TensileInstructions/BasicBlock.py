from typing import List, Type, Tuple
from itertools import dropwhile, takewhile
from .Code import Label, Module
from .Instructions import MFMAInstruction, Instruction, CommonInstruction, ReadWriteInstruction,\
    RegisterContainer, LocalWriteInstruction, GlobalReadInstruction, SWaitCnt, VMovB32, VMovB64,\
    SBarrier, SSetPrior

class BasicBlock:
    def __init__(self, mod: Module):
        self.instructions = mod.flatitems()

    def iterInstType(self, t: Type):
        return filter(lambda x: isinstance(x, t), self.instructions)

    def toModule(self):
        mod = Module()
        for inst in self.instructions:
            mod.add(inst)
        return mod

def hasDependency(ins0: CommonInstruction, ins1: CommonInstruction):
    '''
    Check if ins1 uses result of inst0, i.e. ins0 -> ins1.
    '''
    srcs: List[RegisterContainer] = []
    
    if isinstance(ins1, CommonInstruction):
        srcs = ins1.srcs 
    elif isinstance(ins1, MFMAInstruction):
        srcs = [ins1.a, ins1.b]
    elif isinstance(ins1, GlobalReadInstruction):
        srcs = [ins1.vaddr]
    elif isinstance(ins1, LocalWriteInstruction):
        srcs = [ins1.src0]

    if isinstance(ins0, LocalWriteInstruction):
        dst: RegisterContainer = ins0.dstAddr
    elif hasattr(ins0, 'dst'):
        dst: RegisterContainer = ins0.dst
    elif isinstance(ins0, SSetPrior):
        return False
    else:
        return True

    return any(dst == s for s in srcs)

def unrolledLoopCvtOptimizer(bb: BasicBlock, miLatency: int):
    class ScheduledMfma:
        def __init__(self, mfma: MFMAInstruction, latencyLeft: int, insts: List[Instruction]):
            self.mfma = mfma
            self.latencyLeft = latencyLeft
            self.instructions = insts

    prologue = [i for i in takewhile(lambda x: not isinstance(x, MFMAInstruction), bb.instructions)]
    Scheduling = List[ScheduledMfma]
    mfmas: Scheduling = []

    def makeReverseSchedulingIter(mfmas: Scheduling):
        for i, mfma in enumerate(reversed(mfmas)):
            for j, inst in enumerate(reversed(mfma.instructions)):
                yield mfma, len(mfmas) - i - 1, inst, len(mfma.instructions) - j - 1

    def makeSchedulingIter(mfmas: Scheduling):
        for i, mfma in enumerate(mfmas):
            for j, inst in enumerate(mfma.instructions):
                yield mfma, i, inst, j

    def calculateSchedulingScore(mfmas: Scheduling, miLatency: int):
        numMfmas = len(mfmas)
        sumExtraLatencies = sum(abs(mfma.latencyLeft) for mfma in mfmas if mfma.latencyLeft < 0)
        return 1 - sumExtraLatencies / (numMfmas * miLatency)

    def getInstLatency(inst: Instruction) -> int:
        if isinstance(inst, ReadWriteInstruction):
            return inst.issueLatency()
        elif isinstance(inst, Instruction):
            return 1
        return 0

    for inst in dropwhile(lambda x: not isinstance(x, MFMAInstruction), bb.instructions):
        if isinstance(inst, MFMAInstruction):
            mfmas.append(ScheduledMfma(inst, miLatency, []))
            continue
        elif isinstance(inst, Instruction):
            mfmas[-1].latencyLeft -= getInstLatency(inst)
            mfmas[-1].instructions.append(inst)
        elif isinstance(inst, Label):
            mfmas[-1].instructions.append(inst)

    print(f'Optimizable # of MFMAs: {sum(mfma.latencyLeft < 0 for mfma in mfmas)}')
    print(f'Scheduling score: {calculateSchedulingScore(mfmas, miLatency)}')

    def optimizeGlobalReads() -> int:
        '''
        Move global read to its corresponding s_waitcnt vm(x) and v_mov as close as possible.
        '''
        def findCorrespondingSWaitOrVMovIdx(srcMfmaIdx: int, mfmas, srcIdx: int) -> Tuple[int, int]:
            dstMfmaIdx = None
            dstInstIdx = None
            srcInst = mfmas[srcMfmaIdx].instructions[srcIdx]

            for _, mfmaIdx, inst, instIdx in makeReverseSchedulingIter(mfmas):
                doCheck = False
                if mfmaIdx == srcMfmaIdx:
                    if instIdx < srcIdx:
                        doCheck = True
                elif mfmaIdx < srcMfmaIdx:
                    doCheck = True

                if doCheck:
                    if isinstance(inst, SWaitCnt) and inst.vmcnt >= 0:
                        dstMfmaIdx, dstInstIdx = mfmaIdx, instIdx
                        break
                    elif isinstance(inst, SBarrier):
                        dstMfmaIdx, dstInstIdx = mfmaIdx, instIdx
                        break
                    elif isinstance(inst, (VMovB32, VMovB64)):
                        if hasDependency(srcInst, inst):
                            dstMfmaIdx, dstInstIdx = mfmaIdx, instIdx
                            break

            return dstMfmaIdx, dstInstIdx

        def backwardFindFarestIndenpendentInst(inst: Instruction, instIdx: int, begMfmaIdx: int, endMfmaIdx: int, endInstIdx: int, mfmas: List):
            '''
           mfma0  mfma1  mfma2 mfma3  mfma4   mfma5
            |------|--*---|------|-----|---o---|----
                   |  w                |   gl
               endMfmaIdx          begMfmaIdx
            <-----------searching backward----------
            '''
            retMfmaIdx, retInstIdx = None, None

            for _, mfmaIdx, i, iIdx in makeReverseSchedulingIter(mfmas):
                if mfmaIdx < endMfmaIdx:
                    break
                elif mfmaIdx > begMfmaIdx:
                    continue
                elif mfmaIdx == begMfmaIdx:
                    if iIdx < instIdx:
                        if not hasDependency(inst, i):
                            retMfmaIdx, retInstIdx = mfmaIdx, iIdx
                        else:
                            break
                elif mfmaIdx == endMfmaIdx:
                    if iIdx > endInstIdx:
                        if not hasDependency(inst, i):
                            retMfmaIdx, retInstIdx = mfmaIdx, iIdx
                        else:
                            break
                    else:
                        break
                else:
                    if not hasDependency(inst, i):
                        retMfmaIdx, retInstIdx = mfmaIdx, iIdx
                    else:
                        break
                #TODO: check dependency of mfma and inst in case of direct to vgpr

            return retMfmaIdx, retInstIdx

        numInstsMoved = 0
        srcMfmaIdx, srcInstIdx, dstMfmaIdx, dstInstIdx = None, None, None, None

        for _, mfmaIdx, inst, instIdx in makeSchedulingIter(mfmas):
            if isinstance(inst, GlobalReadInstruction):
                swaitMfmaIdx, swaitInstIdx = findCorrespondingSWaitOrVMovIdx(mfmaIdx, mfmas, instIdx)

                if (swaitMfmaIdx, swaitInstIdx) == (None, None):
                    #To prevent from moving all global reads to the beginning for loop.
                    continue

                targetMfmaIdx, targetInstIdx = backwardFindFarestIndenpendentInst(inst, instIdx, mfmaIdx, swaitMfmaIdx, swaitInstIdx, mfmas)

                if (targetMfmaIdx, targetInstIdx) == (None, None):
                    continue
                
                srcMfmaIdx, srcInstIdx, dstMfmaIdx, dstInstIdx = mfmaIdx, instIdx, targetMfmaIdx, targetInstIdx
                numInstsMoved += 1
                break

        if (srcMfmaIdx, srcInstIdx, dstMfmaIdx, dstInstIdx) != (None, None, None, None):
            srcInst = mfmas[srcMfmaIdx].instructions.pop(srcInstIdx)
            srcInstLatency = getInstLatency(srcInst)
            mfmas[srcMfmaIdx].latencyLeft += srcInstLatency
            mfmas[dstMfmaIdx].latencyLeft -= srcInstLatency
            mfmas[dstMfmaIdx].instructions.insert(dstInstIdx, srcInst)

        return numInstsMoved

    def tryMakeMoreRoomFromTail():
        startIndices = [i - 1 for i, mfma in enumerate(mfmas) if any(isinstance(j, SBarrier) for j in mfma.instructions)]
        numInstsMoved = 0

        for j in reversed(startIndices):
            for i in reversed(range(1, j)):
                mfma = mfmas[i]
                nonfreeIdx = None

                if mfma.instructions:
                    nonfreeIdx = i
                    break

            if nonfreeIdx is None:
                break

            freeIdx = nonfreeIdx + 1
            instsToMove = []

            for inst in reversed(mfmas[nonfreeIdx].instructions):
                if not hasDependency(inst, mfmas[freeIdx].mfma) and mfmas[freeIdx].latencyLeft > 0:
                    instsToMove.append(inst)
                    latency = getInstLatency(inst)
                    mfmas[freeIdx].latencyLeft -= latency
                    mfmas[nonfreeIdx].latencyLeft += latency
                    numInstsMoved += 1
                else:
                    break

            for _ in instsToMove:
                inst = mfmas[nonfreeIdx].instructions.pop(-1)
                mfmas[freeIdx].instructions.insert(0, inst)

        return numInstsMoved

    def optimize(reverse=False):
        '''
        Minimize the exceeded latency for each scheduled MFMA.
        '''
        numInstsMoved = 0
        begIdx, endIdx = 1, len(mfmas) - 1
        it = range(begIdx, endIdx)

        if reverse:
            it = reversed(it)

        for i in it:
            mfma = mfmas[i]
            prevMfma = mfmas[i - 1]
            nextMfma = mfmas[i + 1]

            if mfma.latencyLeft < 0 and mfma.instructions:
                # if prevMfma.latencyLeft > 0 and not hasDependency(mfma.instructions[0], mfma.mfma):
                #     if isinstance(mfma.instructions[0], (SBarrier,)):
                #         continue

                #     instToMove = mfma.instructions.pop(0)
                #     instLatency = getInstLatency(instToMove)
                #     mfma.latencyLeft += instLatency
                #     prevMfma.instructions.append(instToMove)
                #     prevMfma.latencyLeft -= instLatency
                #     numInstsMoved += 1
                #     print(f'Optimzed: move {str(instToMove).rstrip()} from MFMA-{i} to MFMA-{i-1}')
                #     break
                if nextMfma.latencyLeft >= 0 and not hasDependency(mfma.instructions[-1], nextMfma.mfma):
                    if isinstance(mfma.instructions[-1], (SBarrier, SWaitCnt)):
                        continue
                    instToMove = mfma.instructions.pop(-1)
                    instLatency = getInstLatency(instToMove)
                    mfma.latencyLeft += instLatency
                    nextMfma.instructions.insert(0, instToMove)
                    nextMfma.latencyLeft -= instLatency
                    numInstsMoved += 1
                    print(f'Optimzed: move {str(instToMove).rstrip()} from MFMA-{i} to MFMA-{i+1}')
                    # break
                else:
                    print(f'Unable to optimze MFMA-{i}')

        return numInstsMoved

    print(f'Score before global read optimization: {calculateSchedulingScore(mfmas, miLatency)}')

    while optimizeGlobalReads():
        continue

    print(f'Score after global read optimization: {calculateSchedulingScore(mfmas, miLatency)}')

    prevScore = 0
    curScore = calculateSchedulingScore(mfmas, miLatency)
    numInstsMoved = 0
    thisMoved = 0
    # tryMakeMoreRoomFromTail()

    def getScheduleHash():
        instructions = []

        for mfma in mfmas:
            instructions.append(mfma.mfma)
            instructions.extend(mfma.instructions)

        return hash(''.join(str(i) for i in instructions))

    history = set()

    while prevScore < curScore or thisMoved > 0:
        prevScore = curScore
        thisMoved = optimize(True)
        numInstsMoved += thisMoved
        curScore = calculateSchedulingScore(mfmas, miLatency)
        h = getScheduleHash()

        if h in history:
            print('Duplicated pattern met, break')
            break
        else:
            history.add(h)

    # history.clear()
    # prevScore = 0
    # while prevScore < curScore or thisMoved > 0:
    #     prevScore = curScore
    #     thisMoved = optimize(False)
    #     numInstsMoved += thisMoved
    #     curScore = calculateSchedulingScore(mfmas, miLatency)
    #     h = getScheduleHash()

    #     if h in history:
    #         print('Duplicated pattern met, break')
    #         break
    #     else:
    #         history.add(h)

    print(f'{numInstsMoved} instructions moved')
    print(f'Score after optimization: {calculateSchedulingScore(mfmas, miLatency)}')

    bb.instructions = [i for i in prologue]

    for mfma in mfmas:
        bb.instructions.append(mfma.mfma)
        bb.instructions.extend(mfma.instructions)

    return bb
