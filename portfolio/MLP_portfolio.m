clear all; close all; clc

%% Read data and compute variables
df                      = readtable('MLP/MLP_optimal_ptf_data.csv');
data                    = table2array( df( :, 2:end ) );
date                    = df( :, 1 );
date                    = datetime( date{ :, 1 }, 'InputFormat', 'yyyy-MM-dd');
riskFreeRate            = data( :, end ) / 100;
returns                 = data( :, 1:3 );
expectedReturns         = data( :, 4:6 );
T                       = length( date );
N                       = size ( returns, 2 );
names                   = ['DIA'; 'SPY'; 'QQQ'];

% The function "repmat( A, n, m )" replicates the matrix A n times
% vectically and m times horizontally, so the following line replicates the
% vector of risk-free rates N times horizontally so that its has the same
% dimension as the matrix "returns" and so that we can compute the difference.
excessReturns           = returns - repmat( riskFreeRate, 1, N );
excessExpectedReturns   = expectedReturns - repmat( riskFreeRate, 1, N );
%% Parameters
riskAversion    = 5; % 3 -> risk seeking 7 -> risk averse
nbRiskAversion  = length( riskAversion );

%% a) date > 01/01/2014
% Look-back window to estimate expected returns and covariance matrices
nbWindow        = 120;

% Find the month when we begin our exercice.
indStart    = find( year( date ) == 2014 & month( date ) == 1 & day( date ) == 2 );

% Strategies
% When surrounded by bracktest (i.e. "{" and "}"), we create a matrix of
% cells. Cells are useful because they can contain several types of
% variables (words, numbers, etc.)
strategyNames   = { 'Sample Average Excess Return'; ...
                    'NN Average Excess Return'; ...
                    'Sample Average Excess Return (no short-selling)'; ...
                    'NN Average Excess Return (no short-selling)'; ...
                    'Sample Average Excess Return DCC'; ...
                    'NN Average Excess Return DCC'; ...
                    'Sample Average Excess Return (no short-selling) DCC'; ...
                    'NN Average Excess Return (no short-selling) DCC'; ...
                    'Sample Average Excess Return ADCC'; ...
                    'NN Average Excess Return ADCC'; ...
                    'Sample Average Excess Return (no short-selling) ADCC'; ...
                    'NN Average Excess Return (no short-selling) ADCC'; ...
                    };
nbStrategies = length( strategyNames );

%% Memory allocation
% I pre-allocate different matrices with Not-A-Numbers (NaN). NaNs are
% useful because you can easily spot errors.
optimalAllocation       = cell( nbStrategies, 1 );
for ii = 1:nbStrategies
    % For every strategy, I allocate memory of the matrix of optimal
    % allocations T-by-N.
    optimalAllocation{ ii } = NaN( T, N );
end
portfolioReturns        = NaN( T, nbStrategies );

%% For loop of all the days from January 2011 to April 2015
for t = indStart:T
    
   if week( date( t ) ) == week( date( t - 1 ) )
   % if year( date( t ) ) == year( date( t - 1) )
        % We are not on the first day of the week, so we continue without
        % rebalancing
        continue
   else
        % We are on the first day of the week, we rebalance
        disp( ['We rebalance on ', datestr( date( t ) )] )
   end
   
   % Index of returns used for estimation
   indRollingWindowSample  = t - nbWindow:t - 1;
   
   % Estimate sample moments
   % Compute the sample average excess return and variance on the sample of returns available (up to week t - 1)
   rollingWindowSampleAverageExcessReturns = mean( excessReturns( indRollingWindowSample, : ), 'omitnan' )';
   rollingWindowSampleCovariance           = cov( excessReturns( indRollingWindowSample, : ), 'omitrows' );
   NNAverageExcessReturn = excessExpectedReturns( t, : )';
   [PARAMETERS,LL,HT,VCV,SCORES,DIAGNOSTICS] = dcc(rmmissing ( excessReturns( indRollingWindowSample, : ) ),[],1,0,1);
   [PARAMETERSbis,LLbis,HTbis,VCVbis,SCORESbis,DIAGNOSTICSbis] = dcc(rmmissing ( excessReturns( indRollingWindowSample, : ) ),[],1,1,1);
   
   % Compute optimal allocations
    for strategyIndex   = 1:nbStrategies
        % The switch statement is useful because it executes only the code in the right
        % "case" depending on the strategy name
        switch strategyNames{ strategyIndex }
            case 'Sample Average Excess Return'
                % Allocation of the week using the sample Average Excess Return
                optimalAllocation{ strategyIndex }( t, : )   = inv(rollingWindowSampleCovariance) * rollingWindowSampleAverageExcessReturns / riskAversion;
                
            case 'NN Average Excess Return'
                % Allocation of the week using the NN Average Excess Return
                optimalAllocation{ strategyIndex }( t, : )   = inv(rollingWindowSampleCovariance) * NNAverageExcessReturn / riskAversion;
            
            case 'Sample Average Excess Return (no short-selling)'
                % Allocation of the week using the sample Average Excess Return (no short-selling)
                allocation = inv(rollingWindowSampleCovariance) * rollingWindowSampleAverageExcessReturns / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                 
            case 'NN Average Excess Return (no short-selling)'
                % Allocation of the week using the NN Average Excess Return (no short selling)
                allocation = inv(rollingWindowSampleCovariance) * NNAverageExcessReturn / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                
            case 'Sample Average Excess Return DCC'
                % Allocation of the week using the sample Average Excess Return and DCC
                optimalAllocation{ strategyIndex }( t, : )   = inv( HT(:, :, end) ) * rollingWindowSampleAverageExcessReturns / riskAversion;
                
            case 'NN Average Excess Return DCC'
                % Allocation of the week using the NN Average Excess Return and DCC
                optimalAllocation{ strategyIndex }( t, : )   = inv( HT(:, :, end) ) * NNAverageExcessReturn / riskAversion;
            
            case 'Sample Average Excess Return (no short-selling) DCC'
                % sample Average Excess Return (no short-selling)and DCC
                allocation = inv( HT(:, :, end) ) * rollingWindowSampleAverageExcessReturns / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                 
            case 'NN Average Excess Return (no short-selling) DCC'
                % NN Average Excess Return (no short selling) and DCC
                allocation = inv( HT(:, :, end) ) * NNAverageExcessReturn / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                
            case 'Sample Average Excess Return ADCC'
                % Allocation of the week using the sample Average Excess Return and DCC
                optimalAllocation{ strategyIndex }( t, : )   = inv( HTbis(:, :, end) ) * rollingWindowSampleAverageExcessReturns / riskAversion;
                
            case 'NN Average Excess Return ADCC'
                % Allocation of the week using the NN Average Excess Return and DCC
                optimalAllocation{ strategyIndex }( t, : )   = inv( HTbis(:, :, end) ) * NNAverageExcessReturn / riskAversion;
            
            case 'Sample Average Excess Return (no short-selling) ADCC'
                % sample Average Excess Return (no short-selling)and DCC
                allocation = inv( HTbis(:, :, end) ) * rollingWindowSampleAverageExcessReturns / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                 
            case 'NN Average Excess Return (no short-selling) ADCC'
                % NN Average Excess Return (no short selling) and DCC
                allocation = inv( HTbis(:, :, end) ) * NNAverageExcessReturn / riskAversion;
                allocation( allocation < 0 ) = 0;
                optimalAllocation{ strategyIndex }( t, : )   = allocation;
                
        end
        
        % Compute portfolio returns
        portfolioReturns( t, strategyIndex )    = riskFreeRate( t ) + excessReturns( t, : ) * optimalAllocation{ strategyIndex }( t, : )';
        
    end
    
end


%% Results
indOutOfSample  = indStart:T;

datestr( date( indOutOfSample( 1 ) ) )
datestr( date( indOutOfSample( end ) ) )

% Factor to annualize daily statistics to annual statistics
annualFactor = 52; %52 weeks

R               = portfolioReturns( indOutOfSample, : );
excessR         = R - repmat( riskFreeRate( indOutOfSample ), 1, size( R, 2 ) );

downside = min(0, R);
downside (downside == 0 ) = NaN; % replace all 0 by NaN
downsideVol = sqrt( annualFactor ) * std( downside, 'omitnan');

cumulativeReturn = cumprod( 1 + R/100, 'omitnan');
cumulativeReturn( cumulativeReturn < 0 ) = 0.001;

statisticNames      = { 'Annualized average Excess Returns          '; ...
                        'Annualized volatility                      '; ...
                        'Annualized Sharpe ratio                    '; ...
                        'Annualized Sortino ratio                   '; ...
                        'Max drawdown (%)                           '; ...
                        };
nbStatistics        = length( statisticNames );
statistics          = [];
for ii = 1:nbStatistics
    switch statisticNames{ ii }
        case 'Annualized average Excess Returns          '
            statistics  = [statistics, annualFactor * mean( excessR, 'omitnan' )'];
        case 'Annualized volatility                      '
            statistics  = [statistics, sqrt( annualFactor ) * std( R , 'omitnan' )'];
        case 'Annualized Sharpe ratio                    '
            statistics  = [statistics, sqrt( annualFactor ) * mean( excessR,'omitnan' )' ./ std( R, 'omitnan' )'];
        case 'Annualized Sortino ratio                   '
            statistics  = [statistics, sqrt( annualFactor ) * mean( excessR,'omitnan' )' ./ downsideVol'];
        case 'Max drawdown (%)                           '
            statistics  = [statistics, 100 * maxdrawdown(cumulativeReturn)'];
    
    end
    
end

fprintf( 'Mean-Variance optimization' )
fprintf( '\n' )
fprintf( '\n' )
for strategyIndex   = 1:nbStrategies
    
    fprintf( '%s', [strategyNames{ strategyIndex }, repmat( ' ', 1, 50 - length( strategyNames{ strategyIndex } ) )] )
    for ii = 1:nbStatistics
        fprintf( '\t %5.2f', statistics( strategyIndex, ii ) )
    end
    fprintf( '\n' )
    
end
