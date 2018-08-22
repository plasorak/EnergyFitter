



void Dumper() {

  TFile *f = new TFile("AlexOutput.root", "READ");
  int ChanWidth,Type,NHits,Config,Event;
  float SumADC,TimeWidth;
  std::vector<double>* ENu     = NULL;
  std::vector<double>* HitSADC = NULL;  
  std::vector<double>* HitRMS  = NULL;  

  TTree* tree = (TTree*)f->Get("ClusteredWireHit");
  tree->SetBranchAddress("Config",    &Config   );
  tree->SetBranchAddress("ChanWidth", &ChanWidth);
  tree->SetBranchAddress("Event",     &Event    );
  tree->SetBranchAddress("Type",      &Type     );
  tree->SetBranchAddress("NHits",     &NHits    );
  tree->SetBranchAddress("SumADC",    &SumADC   );
  tree->SetBranchAddress("TimeWidth", &TimeWidth);
  tree->SetBranchAddress("HitSADC",   &HitSADC  );
  tree->SetBranchAddress("HitRMS",    &HitRMS   );

  TTree* truetree = (TTree*)f->Get("TrueInfo");
  truetree->SetBranchAddress("Event", &Event);
  truetree->SetBranchAddress("ENu",   &ENu  );
  
  std::map<int,double> map_evnumber_enu;
  
  for(int j=0;j<truetree->GetEntries();++j) {
    truetree->GetEntry(j);
    map_evnumber_enu[Event] = ENu->front();
  }

  double binning_rms[3] = {0, 2, 1000};
  double binning_adc[4] = {0,100,400,100000};
  TH1D* RMS = new TH1D("RMS","",2,binning_rms);
  TH1D* ADC = new TH1D("ADC","",3,binning_adc);
  
  ofstream myfile;
  for(int i=0; i<6; ++i) {
    std::cout << "Doing config " << i << std::endl;
    int nSign=0;
    int nBack=0;
    myfile.open(Form("Config%i_Binned.txt", i));
    myfile << "Event,ENu,ChanWidth,Type,TimeWidth,NHits,SumADC,RMS0,RMS1,SADC0,SADC1,SADC2"<<std::endl;

    for(int j=0;j<tree->GetEntries();++j) {
      if(j%100000==0)
        std::cout << "did " << 100.*j/tree->GetEntries() << std::endl;
      tree->GetEntry(j);
      if(Config!=i)
        continue;
      ADC->Reset();
      RMS->Reset();
      for (auto const it: (*HitSADC))ADC->Fill(it);
      for (auto const it: (*HitRMS ))RMS->Fill(it);

      myfile << Event <<","
             << map_evnumber_enu[Event]*1000 << ","
             << ChanWidth << ","
             << Type      << ","
             << TimeWidth << ","
             << NHits     << ","
             << SumADC    << ","
             << RMS->GetBinContent(1) << ","
             << RMS->GetBinContent(2) << ","
             << ADC->GetBinContent(1) << ","
             << ADC->GetBinContent(2) << ","
             << ADC->GetBinContent(3) << std::endl;
      nSign+=Type;
      nBack++;
    }
    nBack-=nSign;
    myfile.close();
    std::cout << "nSign " << nSign << std::endl;
    std::cout << "nBack " << nBack << std::endl;

  }
}


